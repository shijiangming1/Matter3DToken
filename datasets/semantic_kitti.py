# Copyright 2026 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
import fcntl
import json
from glob import glob
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

import utils.transforms as tr

from .pc_dataset import PCDataset


class InstanceCutMix:
    def __init__(
        self,
        phase="train",
        temp_dir=None,
        num_instances=40,
    ):
        if temp_dir is None:
            temp_dir = os.environ.get("VAVIT_INSTANCE_CACHE_DIR")
        if temp_dir is None:
            temp_dir = Path(__file__).resolve().parents[1] / "data" / "kitti_instances"

        # Train or Trainval
        self.phase = phase
        assert self.phase in ["train", "trainval"]

        # List of files containing instances for bicycle, motorcycle, person, bicyclist
        self.bank = {1: [], 2: [], 4: [], 5: [], 6: []}

        # Directory where to store instances
        self.rootdir = os.path.join(temp_dir, self.phase)
        self.complete_file = os.path.join(self.rootdir, ".complete")
        self.progress_file = os.path.join(self.rootdir, ".extract_progress.json")
        for id_class in self.bank.keys():
            os.makedirs(os.path.join(self.rootdir, f"{id_class}"), exist_ok=True)

        # Load instances
        for key in self.bank.keys():
            self.bank[key] = glob(os.path.join(self.rootdir, f"{key}", "*.bin"))
        self.__loaded__ = self.test_loaded()
        if not self.__loaded__:
            warnings.warn(
                "Instances must be extracted and saved on disk before training"
            )

        # Augmentations applied on Instances
        self.rot = tr.Compose(
            (
                tr.Flip(inplace=True),
                tr.Rotation(inplace=True),
                tr.Scale(dims=(0, 1, 2), range=0.1, inplace=True),
            )
        )

        # For each class, maximum number of instance to add
        self.num_to_add = num_instances
        print("num instances", num_instances)

        # Voxelization of 1m to downsample point cloud to ensure that
        # center of the instances are at least 1m away
        self.vox = tr.Voxelize(dims=(0, 1, 2), voxel_size=1.0, random=True)

    def test_loaded(self):
        self.__loaded__ = False
        expected = {
            "train": {1: 5083, 2: 3092, 4: 7419, 5: 8084, 6: 1551},
            "trainval": {1: 8213, 2: 4169, 4: 10516, 5: 12190, 6: 2943},
        }[self.phase]

        # A completed local cache is authoritative. Counts can differ between
        # SemanticKITTI releases while still containing all required classes.
        if os.path.isfile(self.complete_file) and all(self.bank[key] for key in self.bank):
            self.__loaded__ = True
            return True

        # Accept a complete cache created by the older implementation, which
        # did not write a completion marker. A genuinely partial 39% cache is
        # rejected and rebuilt instead of being used as training data.
        if all(len(self.bank[key]) == expected[key] for key in self.bank):
            self.__loaded__ = True
            return True
        if all(len(self.bank[key]) >= int(expected[key] * 0.9) for key in self.bank):
            self.__loaded__ = True
            return True

        return False

    def mark_complete(self):
        Path(self.complete_file).touch()

    def prepare_resume(self):
        """Resume extraction without duplicating objects after an interruption."""
        if os.path.isfile(self.progress_file):
            with open(self.progress_file) as progress_file:
                state = json.load(progress_file)
            for key, count in state["counts"].items():
                files = sorted(self.bank[int(key)])
                for pathfile in files[int(count) :]:
                    os.remove(pathfile)
                self.bank[int(key)] = files[: int(count)]
            return int(state["next_index"])

        # Partial caches created by older versions have no safe resume point.
        for key in self.bank:
            for pathfile in self.bank[key]:
                os.remove(pathfile)
            self.bank[key] = []
        return 0

    def save_progress(self, next_index):
        state = {
            "next_index": next_index,
            "counts": {str(key): len(files) for key, files in self.bank.items()},
        }
        temp_file = self.progress_file + ".tmp"
        with open(temp_file, "w") as progress_file:
            json.dump(state, progress_file)
            progress_file.flush()
            os.fsync(progress_file.fileno())
        os.replace(temp_file, self.progress_file)

    def cut(self, pc, class_label, instance_label):
        for id_class in self.bank.keys():
            where_class = class_label == id_class
            all_instances = np.unique(instance_label[where_class])
            for id_instance in all_instances:
                # Segment instance
                where_ins = instance_label == id_instance
                if where_ins.sum() <= 5:
                    continue
                instance = pc[where_ins, :]
                # Center instance
                instance[:, :2] -= instance[:, :2].mean(0, keepdims=True)
                instance[:, 2] -= instance[:, 2].min(0, keepdims=True)
                # Save instance
                pathfile = os.path.join(
                    self.rootdir, f"{id_class}", f"{len(self.bank[id_class]):07d}.bin"
                )
                instance.tofile(pathfile)
                self.bank[id_class].append(pathfile)

    def mix(self, pc, class_label):
        # Find potential location where to add new object (on a surface)
        pc_vox, class_label_vox = self.vox(pc, class_label)
        where_surface = np.where((class_label_vox >= 8) & (class_label_vox <= 10))[0]
        where_surface = where_surface[torch.randperm(len(where_surface))]

        # Add instances of each class in bank
        id_tot = 0
        new_pc, new_label = [pc], [class_label]
        for id_class in self.bank.keys():
            nb_to_add = torch.randint(self.num_to_add, (1,))[0]
            nb_to_add = np.min((nb_to_add, len(where_surface) - id_tot))
            which_one = torch.randint(len(self.bank[id_class]), (nb_to_add,))
            for ii in range(nb_to_add):
                # Point p where to add the instance
                p = pc_vox[where_surface[id_tot]]
                # Extract instance
                object = self.bank[id_class][which_one[ii]]
                object = np.fromfile(object, dtype=np.float32).reshape((-1, 4))
                # Augment instance
                label = np.ones((object.shape[0],), dtype=np.int32) * id_class
                object, label = self.rot(object, label)
                # Move instance at point p
                object[:, :3] += p[:3][None]
                # Add instance in the point cloud
                new_pc.append(object)
                # Add corresponding label
                new_label.append(label)
                id_tot += 1

        return np.concatenate(new_pc, 0), np.concatenate(new_label, 0)

    def __call__(self, pc, class_label, instance_label):
        if not self.__loaded__:
            self.cut(pc, class_label, instance_label)
            return None, None
        return self.mix(pc, class_label)


class SemanticKITTI(PCDataset):
    CLASS_NAME = [
        [
            "car",  # 0
            "bicycle",  # 1
            "motorcycle",  # 2
            "truck",  # 3
            "other-vehicle",  # 4
            "person",  # 5
            "bicyclist",  # 6
            "motorcyclist",  # 7
            "road",  # 8
            "parking",  # 9
            "sidewalk",  # 10
            "other-ground",  # 11
            "building",  # 12
            "fence",  # 13
            "vegetation",  # 14
            "trunk",  # 15
            "terrain",  # 16
            "pole",  # 17
            "traffic-sign",  # 18
        ]
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Config file and class mapping
        current_folder = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_folder, "semantic-kitti.yaml")) as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml["learning_map"]

        # Split
        if self.phase == "train":
            split = semkittiyaml["split"]["train"]
        elif self.phase == "val":
            split = semkittiyaml["split"]["valid"]
        else:
            raise Exception(f"Unknown split {self.phase}")

        # Find all files
        self.im_idx = []
        for i_folder in np.sort(split):
            self.im_idx.extend(
                glob(
                    os.path.join(
                        self.rootdir,
                        "dataset",
                        "sequences",
                        str(i_folder).zfill(2),
                        "velodyne",
                        "*.bin",
                    )
                )
            )
        self.im_idx = np.sort(self.im_idx)

        # Training with instance cutmix
        if self.instance_cutmix:
            # CutMix
            print("Apply cutmix")
            assert self.phase != "test" and self.phase != "val", (
                "Instance cutmix should not be applied at test or val time"
            )
            self.cutmix = InstanceCutMix(
                phase=self.phase,
                temp_dir=self.instance_cache_dir,
                num_instances=self.num_instances,
            )
            if not self.cutmix.test_loaded():
                if not self.prepare_instance_cache:
                    raise RuntimeError(
                        "SemanticKITTI InstanceCutMix cache is incomplete. "
                        "Run train.py once with --prepare_instance_cache."
                    )
                lock_path = os.path.join(
                    os.path.dirname(self.cutmix.rootdir), ".extract.lock"
                )
                with open(lock_path, "a+") as lock_file:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    # Another process may have completed the bank while we waited.
                    for key in self.cutmix.bank:
                        self.cutmix.bank[key] = glob(
                            os.path.join(self.cutmix.rootdir, f"{key}", "*.bin")
                        )
                    if not self.cutmix.test_loaded():
                        print("Extracting instances before training...")
                        start_index = self.cutmix.prepare_resume()
                        for index in tqdm(range(start_index, len(self))):
                            self.load_pc(index)
                            self.cutmix.save_progress(index + 1)
                        self.cutmix.mark_complete()
                        if os.path.isfile(self.cutmix.progress_file):
                            os.remove(self.cutmix.progress_file)
                        print("Done.")
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            elif not os.path.isfile(self.cutmix.complete_file):
                self.cutmix.mark_complete()
            assert self.cutmix.test_loaded(), "Instances not extracted correctly"

    def __len__(self):
        return len(self.im_idx)

    def __load_pc_internal__(self, index):
        # Load point cloud
        pc = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))

        # Extract Label
        labels_inst = np.fromfile(
            self.im_idx[index].replace("velodyne", "labels")[:-3] + "label",
            dtype=np.uint32,
        ).reshape((-1, 1))
        labels = labels_inst & 0xFFFF  # delete high 16 digits binary
        labels = np.vectorize(self.learning_map.__getitem__)(labels).astype(np.int32)

        # Map ignore index 0 to 255
        labels = labels[:, 0] - 1
        labels[labels == -1] = 255

        return pc, labels, labels_inst[:, 0]

    def load_pc(self, index):
        pc, labels, labels_inst = self.__load_pc_internal__(index)

        # Instance CutMix and Polarmix
        if self.instance_cutmix:
            pc, labels = self.cutmix(pc, labels, labels_inst)

        return pc, labels, self.im_idx[index]
