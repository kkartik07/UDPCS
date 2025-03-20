"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from typing import Optional
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS, default_loader
from torchvision.datasets.utils import download_and_extract_archive
from ._util import check_exits

#
import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits



class OfficeCaltech(ImageList):
    """Office+Caltech Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr, ``'W'``:webcam and ``'C'``: caltech.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            amazon/
                images/
                    backpack/
                        *.jpg
                        ...
            dslr/
            webcam/
            caltech/
            image_list/
                amazon.txt
                dslr.txt
                webcam.txt
                caltech.txt
    """
    # directories = {
    #     "A": "amazon",
    #     "D": "dslr",
    #     "W": "webcam",
    #     "C": "caltech10"
    # }
    download_list = [
        ("amazon", "amazon", "https://github.com/ZhangJUJU/OfficeCaltechDomainAdaptation/tree/master/images/amazon"),
        ("caltech10", "caltech10", "https://github.com/ZhangJUJU/OfficeCaltechDomainAdaptation/tree/master/images/caltech10"),
        ("dslr", "dslr", "https://github.com/ZhangJUJU/OfficeCaltechDomainAdaptation/tree/master/images/dslr"),
        ("webcam", "webcam", "https://github.com/ZhangJUJU/OfficeCaltechDomainAdaptation/tree/master/images/webcam"),
        
    ]
    image_list = {
        "a": "amazon",
        "c": "caltech10",
        "d": "dslr",
        "w": "webcam",
        
    }
    CLASSES = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
               'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']

    # def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):
    #     if download:
    #         for dir in self.directories.values():
    #             if not os.path.exists(os.path.join(root, dir)):
    #                 download_and_extract_archive(url="https://github.com/ZhangJUJU/OfficeCaltechDomainAdaptation/tree/master/images",
    #                                              download_root=os.path.join(root, 'download'),
    #                                              filename="officecaltech.gz", remove_finished=False, extract_root=root)
    #                 break
    #     else:
    #         list(map(lambda dir, _: check_exits(root, dir), self.directories.values()))

    #     super(OfficeCaltech, self).__init__(
    #         os.path.join(root, self.directories[task]), default_loader, extensions=IMG_EXTENSIONS, **kwargs)
    #     self.classes = OfficeCaltech.CLASSES
    #     self.class_to_idx = {cls: idx
    #                          for idx, clss in enumerate(self.classes)
    #                          for cls in clss}

    # @property
    # def num_classes(self):
    #     """Number of classes"""
    #     return len(self.classes)

    # @classmethod
    # def domains(cls):
    #     return list(cls.directories.keys())

    def __init__(self, root: str, task: str, split: Optional[str] = 'train', download: Optional[float] = False, **kwargs):
        assert task in self.image_list
        # assert split in ['train', 'test']
        data_list_file = os.path.join(root, "image_list", "{}.txt".format(self.image_list[task], split))
        print("loading {}".format(data_list_file))

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda args: check_exits(root, args[0]), self.download_list))

        super(OfficeCaltech, self).__init__(root, OfficeCaltech.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
