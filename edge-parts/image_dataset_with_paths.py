import torchvision


class ImageDatasetWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes filepath, image, and target."""

    # Idea from
    # https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d.
    def __getitem__(self, index):
        img, label = super(ImageDatasetWithPaths, self).__getitem__(index)
        fpath = self.imgs[index][0]
        return fpath, img, label
