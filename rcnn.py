import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class TrafficSignsDataset(torch.utils.data.Dataset):
    def __init__(self, data, image_dir, transforms=None):
        self.data = data
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        image_path = self.image_dir + record['Path']
        image = Image.open(image_path).convert('RGB')
        boxes = torch.tensor([[record['Roi.X1'], record['Roi.Y1'], record['Roi.X2'], record['Roi.Y2']]],
                             dtype=torch.float32)
        labels = torch.tensor([record['ClassId']], dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, [target]  # Wrap target in a list

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform():
    def transform(image, target):
        image = F.to_tensor(image)
        return image, target
    return transform

def main():
    df = pd.read_csv(r'C:/Users/spika/Desktop/ACG/MLA/Project/Traffic_Sign_Dataset/Traffic_Sign_Dataset/Train.csv')
    dataset = TrafficSignsDataset(df, r'C:/Users/spika/Desktop/ACG/MLA/Project/Traffic_Sign_Dataset/Traffic_Sign_Dataset/', get_transform())
    data_loader = DataLoader(dataset, batch_size=120, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(torch.cuda.is_available())
    # print(torch.cuda.current_device())

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 43  # Including background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 1

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        for i, (images, targets) in enumerate(data_loader):
            print(f"Iteration {i + 1}/{len(data_loader)}")
            print(f"Targets at iteration {i}: {targets}")
            images = list(image.to(device) for image in images)

            # Process targets for each image in the batch
            batch_targets = []
            for target_list in targets:
                target_dict = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for target in target_list
                    for k, v in target.items()
                }
                batch_targets.append(target_dict)

            # Forward pass
            loss_dict = model(images, batch_targets)  # Returns losses and detections

            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

    # Save the model
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Model saved.")


if __name__ == "__main__":
    main()
