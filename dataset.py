import pickle
from torch_geometric.data import Dataset


class MyDataset(Dataset):
    def __init__(self, *args):
        super().__init__()
        if args[0] == "file":
            file_path = args[1]
            with open(file_path, 'rb') as f:
                self.G_list = pickle.load(f)
            self.len = len(self.G_list)
        elif args[0] == "list":
            self.G_list = args[1]
            self.len = len(args[1])

    def __getitem__(self, index):
        G = self.G_list[index]
        return G[0], G[1], G[2]

    def __len__(self):
        return self.len

    def get_by_idx(self, idx):
        return [self.G_list[i] for i in idx]


if __name__ == '__main__':
    data_root = './data'
    dataset = MyDataset("file", "./data/processed/valid_5A.pkl")
