import argparse
import json
import os
from dataclasses import dataclass
from typing import *

from numpy.typing import NDArray
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from tqdm import tqdm


@dataclass
class Metadata:
    patient_id: str
    age: int
    prostate_specific_antigen_level: float

    @classmethod
    def from_json(cls, path: str) -> 'Metadata':
        with open(path) as f:
            json_object = json.load(f)
        return Metadata(json_object["patient_id"], json_object["age"], json_object["psa"])


class CustomDataset:
    def __init__(self, data_folder = str, max_datapoints: Optional[int] = None) -> None:
        images: List[Nifti1Image] = []
        image_data: List[NDArray] = []
        metadata: List[Metadata] = []
        labels: List[int] = []  # 0 if low grade, 1 if high grade

        num_datapoints = 0
        if not os.path.exists(data_folder):
            raise ValueError(f"Dataset folder does not exist `{data_folder}`")
        
        for folder_name in tqdm(os.listdir(data_folder)):
            folder = os.path.join(data_folder, folder_name)
            if not "case" in folder:
                continue

            image_path = os.path.join(folder, f'{folder_name}.nii.gz')
            if not os.path.isfile(image_path):
                raise ValueError(f"Missing image file `{image_path}`")
            images.append(nib.load(image_path))
            image_data.append(images[-1].get_fdata())

            metadata_path = os.path.join(folder, f"{folder_name}.json")
            if not os.path.isfile(metadata_path):
                raise ValueError(f"Missing metadata file `{metadata_path}`")
            metadata.append(Metadata.from_json(path=metadata_path))

            ground_truth_path = os.path.join(folder, f"{folder_name}_ground_truth.json")
            if not os.path.isfile(ground_truth_path):
                raise ValueError(f"Missing ground truth file `{ground_truth_path}`")
            with open(ground_truth_path) as f:
                labels.append(int(f.read().lower().strip().replace("\"", "") == "high"))

            num_datapoints += 1
            if max_datapoints is not None and num_datapoints > max_datapoints:
                break

        if len(images) == 0:
            raise ValueError(f"No datapoints found in folder `{data_folder}`")

        self.images = images
        self.image_data = image_data
        self.metadata = metadata
        self.labels = labels

        assert len(self.images) == len(self.metadata) == len(self.labels)

        print(f"loaded {self}")

    def __repr__(self) -> str:   
        num_true_labels = sum(self.labels)
        average_age = sum([m.age for m in self.metadata]) / len(self.metadata)
        return f"Dataset with {len(self.images)} images, {num_true_labels} of which are high grade. Average age: {average_age:.0f}"

    def __getitem__(self, index: int) -> Tuple[Nifti1Image, Metadata, int]:
        return self.images[index], self.metadata[index], self.labels[index]

    def __len__(self):
        return len(self.images)

    def plot_age_distribution(self) -> str:
        ages = [m.age for m in self.metadata]
        plt.hist(ages)
        plt.xlabel("Age")
        plt.show()

    def plot_psa_distribution(self) -> str:
        psa = [m.prostate_specific_antigen_level for m in self.metadata]
        plt.hist(psa)
        plt.xlabel( "PSA")
        plt.show()

    def plot_example_images(self) -> str:
        num_images = 10
        for i in range(num_images):
            plt.subplot(2, num_images // 2, i + 1)
            plt.imshow(self.image_data[i][:, :, 0], cmap='bone')
            plt.axis('off')
        plt.show()

    def plot_single_patient_images(self) -> str:
        data = self.image_data[3]
        num_images = data.shape[-1]
        for i in range(num_images):
            plt.subplot(2, num_images // 2, i + 1)
            plt.imshow(data[:, :, i], cmap='bone')
            plt.axis('off')
        plt.show()

    def plot_test_images(self) -> str:
        num_images = 15
        num_patients = 5
        plt.figure(figsize=(15, 5))
        count = 0
        labels = dict()
        for i in range(len(self)):
            labels[self.labels[i]] = labels.get(self.labels[i], 0) + 1
            if labels[self.labels[i]] > num_patients // 2:  # only two patients per label
                continue
            for j in range(num_images):
                plt.subplot(4, num_images, j + 1 + count * num_images)
                plt.imshow(self.image_data[i][:, :, j], cmap='bone' if self.labels[i] == 1 else 'gray')
                plt.axis('off')            
            count += 1
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-folder', type=str, required=True)
    parser.add_argument('-pa', '--plot-age', action='store_true')
    parser.add_argument('-pp', '--plot-psa', action='store_true')
    parser.add_argument('-pi', '--plot-example-images', action='store_true')
    parser.add_argument('-psp', '--plot-single-patient-images', action='store_true')
    args = parser.parse_args()

    dataset = CustomDataset(args.data_folder)

    if args.plot_age:
        dataset.plot_age_distribution()
    if args.plot_psa:
        dataset.plot_psa_distribution()
    if args.plot_example_images:
        dataset.plot_example_images()
    if args.plot_single_patient_images:
        dataset.plot_single_patient_images()

    dataset.plot_test_images()
    