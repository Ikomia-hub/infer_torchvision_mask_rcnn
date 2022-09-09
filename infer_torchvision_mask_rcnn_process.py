from ikomia import core, dataprocess
from ikomia.dnn.torch import models
import os
import copy
import torch
import torchvision.transforms as transforms
import random


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class MaskRcnnParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = 'MaskRcnn'
        self.dataset = 'Coco2017'
        self.model_path = ''
        self.classes_path = os.path.dirname(os.path.realpath(__file__)) + "/models/coco2017_classes.txt"
        self.confidence = 0.5
        self.mask_threshold = 0.5
        self.update = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.dataset = param_map["dataset"]
        self.model_path = param_map["model_path"]
        self.classes_path = param_map["classes_path"]
        self.confidence = float(param_map["confidence"])
        self.mask_threshold = float(param_map["mask_threshold"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["model_name"] = self.model_name
        param_map["dataset"] = self.dataset
        param_map["model_path"] = self.model_path
        param_map["classes_path"] = self.classes_path
        param_map["confidence"] = str(self.confidence)
        param_map["mask_threshold"] = str(self.mask_threshold)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class MaskRcnn(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.model = None
        self.class_names = []
        self.colors = []
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Remove graphics input
        self.removeInput(1)
        # Add instance segmentation output
        self.addOutput(dataprocess.CInstanceSegIO())

        # Create parameters class
        if param is None:
            self.setParam(MaskRcnnParam())
        else:
            self.setParam(copy.deepcopy(param))

    def load_class_names(self):
        self.class_names.clear()
        param = self.getParam()

        with open(param.classes_path) as f:
            for row in f:
                self.class_names.append(row[:-1])

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 3

    def predict(self, image):
        trs = transforms.Compose([
            transforms.ToTensor(),
            ])

        input_tensor = trs(image)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        return output

    def generate_colors(self):
        # we use seed to keep the same color for our boxes + labels (same random each time)
        self.colors.append([0, 0, 0])
        random.seed(10)

        for cl in self.class_names:
            self.colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255])

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get parameters :
        param = self.getParam()

        # Get input :
        img_input = self.getInput(0)
        src_image = img_input.getImage()
        h, w, _ = src_image.shape

        # Step progress bar:
        self.emitStepProgress()

        # Load model
        if self.model is None or param.update:
            # Load class names
            self.load_class_names()
            # Load model
            use_torchvision = param.dataset != "Custom"
            self.model = models.mask_rcnn(use_pretrained=use_torchvision, classes=len(self.class_names))
            if param.dataset == "Custom":
                self.model.load_state_dict(torch.load(param.model_path, map_location=self.device))

            self.model.to(self.device)
            self.generate_colors()
            param.update = False

        pred = self.predict(src_image)
        cpu = torch.device("cpu")
        boxes = pred[0]["boxes"].to(cpu).numpy().tolist()
        scores = pred[0]["scores"].to(cpu).numpy().tolist()
        labels = pred[0]["labels"].to(cpu).numpy().tolist()
        masks = pred[0]["masks"]

        # Step progress bar:
        self.emitStepProgress()

        # Forward input image to result image
        self.forwardInputImage(0, 0)

        # Init graphics output
        instance_output = self.getOutput(1)
        instance_output.init("MaskRCNN", 0, w, h)

        # Get predictions
        valid_results = [scores.index(x) for x in scores if x > param.confidence]
        for i in valid_results:
            # box
            box_x = float(boxes[i][0])
            box_y = float(boxes[i][1])
            box_w = float(boxes[i][2] - boxes[i][0])
            box_h = float(boxes[i][3] - boxes[i][1])
            mask = (masks[i] > param.mask_threshold).byte()
            instance_output.addInstance(i, 0, labels[i], self.class_names[labels[i]], float(scores[i]),
                                        box_x, box_y, box_w, box_h,
                                        mask.squeeze().cpu().numpy(), self.colors[labels[i] + 1])

        self.setOutputColorMap(0, 1, self.colors)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class MaskRcnnFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_torchvision_mask_rcnn"
        self.info.shortDescription = "Mask R-CNN inference model for object detection and segmentation."
        self.info.description = "Mask R-CNN inference model for object detection and segmentation. " \
                                "Implementation from PyTorch torchvision package. " \
                                "This Ikomia plugin can make inference of pre-trained model from " \
                                "COCO dataset or custom trained model. Custom training can be made with " \
                                "the associated MaskRCNNTrain plugin from Ikomia marketplace."
        self.info.authors = "Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick"
        self.info.article = "Mask R-CNN"
        self.info.journal = "Proceedings of the IEEE International Conference on Computer Vision (ICCV)"
        self.info.year = 2017
        self.info.licence = "BSD-3-Clause License"
        self.info.documentationLink = "https://arxiv.org/abs/1703.06870"
        self.info.repository = "https://github.com/pytorch/vision"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.iconPath = "icons/pytorch-logo.png"
        self.info.version = "1.2.0"
        self.info.keywords = "torchvision,detection,segmentation,instance,object,resnet,pytorch"

    def create(self, param=None):
        # Create process object
        return MaskRcnn(self.info.name, param)
