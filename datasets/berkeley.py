import os
import os.path as osp
import zipfile
import requests
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image


class BerkeleyDataset:
    """Berkeley CycleGAN dataset"""

    AVAIL_DATASETS = [
        "ae_photos", "apple2orange", "summer2winter_yosemite",
        "horse2zebra", "monet2photo", "cezanne2photo",
        "ukiyoe2photo", "vangogh2photo", "maps",
        "cityscapes", "facades", "iphone2dslr_flower"
    ]
    BASE_URL = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets"

    def __init__(self, root, name, transform, train=True):
        self.root = root
        self.name = name
        self.transform = transform
        self.train = train

        # Sanity check
        if name not in BerkeleyDataset.AVAIL_DATASETS:
            raise RuntimeError("Unknown dataset '{}'...".format(name))

        # Download dataset
        dataset_url = osp.join(BerkeleyDataset.BASE_URL, name+".zip")
        dataset_dir = osp.join(root, name)
        self._download_dataset(dataset_dir, dataset_url)

        # Extract file names
        dir_name = "train" if train else "test"
        files_A = [ osp.join(dataset_dir, dir_name+"A", f)
                    for f in os.listdir(osp.join(dataset_dir, dir_name+"A")) ]
        files_B = [ osp.join(dataset_dir, dir_name+"B", f)
                    for f in os.listdir(osp.join(dataset_dir, dir_name+"B")) ]

        # Align number of file between A and B
        max_length = max(len(files_A), len(files_B))
        self.files_A = files_A + np.random.choice(files_A, max_length-len(files_A)).tolist()
        self.files_B = files_B + np.random.choice(files_B, max_length-len(files_B)).tolist()

    def __getitem__(self, idx):
        img_A = Image.open(self.files_A[idx]).convert('RGB')
        img_B = Image.open(self.files_B[idx]).convert('RGB')

        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return img_A, img_B

    def __len__(self):
        return len(self.files_A)

    def _download_dataset(self, dataset_dir, dataset_url):
        if osp.exists(dataset_dir):
            return

        # Create dataset diretory & download dataset
        fpath = osp.join(osp.dirname(dataset_dir), osp.basename(dataset_url))
        download_from_url(dataset_url, fpath)

        # Extract zip file
        zip_ = zipfile.ZipFile(fpath, 'r')
        zip_.extractall(osp.dirname(dataset_dir))
        zip_.close()

        # Remove zip file
        os.remove(fpath)

def download_from_url(url, dst):
    """Download file

    Args:
        url (str): url to download file
        dst (str): place to put the file
    """
    file_size = int(requests.head(url).headers['Content-Length'])

    if osp.exists(dst):
        first_byte = osp.getsize(dst)
    else:
        first_byte = 0

    if first_byte >= file_size:
        return file_size

    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}

    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])

    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    dataset = BerkeleyDataset(root="download", name="horse2zebra", transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch_idx, (img_A, img_B) in enumerate(dataloader):
        print(batch_idx, img_A.shape, img_B.shape)
