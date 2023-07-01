# Define the model as before
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = len(train_df['ClassId'].unique())
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
