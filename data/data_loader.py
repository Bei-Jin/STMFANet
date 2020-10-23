from data.base_data_loader import BaseDataLoader
import torch.utils.data
import pdb

def CreateDataset(opt):
    dataset = None
    if opt.data == 'KTH':
        from data.kth_dataset import KthDataset
        dataset = KthDataset()
    else:
        raise ValueError('Dataset [%s] not recognized.' % opt.data)

    print('dataset [%s] was created' % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDataLoader(BaseDataLoader):
    def name(self):
        return 'CreateDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        if opt.debug:
            opt.serial_batches = True
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches, num_workers=int(opt.nThreads), drop_last=True)
    
    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)  
        
def CreateDataLoader(opt):
    data_loader = CustomDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
