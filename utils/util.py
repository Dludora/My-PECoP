import os.path
import pickle

dir_path = '/'.join(os.path.dirname(__file__).split('/')[:-1])


def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    return pkl_data

if __name__ == '__main__':
    train_data = read_pkl('MTL-AQA_split_0_data', 'train_split_0.pkl')
    train_label = read_pkl('MTL-AQA_split_0_data', 'final_annotations_dict.pkl')
    print(train_data)
    print(train_label)
