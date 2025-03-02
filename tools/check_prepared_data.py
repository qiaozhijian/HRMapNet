import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the map
    test_map_file = "/home/qzj/code/HRMapNet/maps/test_map.pt"
    test_map = torch.load(test_map_file, map_location="cpu")
    singapore_map = test_map["singapore-onenorth"].numpy() # [H, W, 3]

    # see if there is any value is between 99 and  130
    print(np.min(singapore_map), np.max(singapore_map))
    print(np.unique(singapore_map))

    # plt.imshow(singapore_map)
    # plt.show()

    train_map_file = "/home/qzj/code/HRMapNet/maps/train_map.pt"
    train_map = torch.load(train_map_file, map_location="cpu")
    singapore_map = train_map["singapore-onenorth"].numpy() # [H, W, 3]
    # plt.imshow(singapore_map)
    # plt.show()

    print(np.min(singapore_map), np.max(singapore_map))
    print(np.unique(singapore_map))