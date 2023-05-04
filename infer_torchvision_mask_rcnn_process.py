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
        self.model_name_or_path = ""
        self.model_name = 'MaskRcnn'
        self.dataset = 'Coco2017'
        self.model_path = ''
        self.classes_path = os.path.dirname(os.path.realpath(__file__)) + "/models/coco2017_classes.txt"
        self.conf_thres = 0.5
        self.iou_thres = 0.5
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name_or_path = param_map["model_name_or_path"]
        self.model_name = param_map["model_name"]
        self.dataset = param_map["dataset"]
        self.model_path = param_map["model_path"]
        self.classes_path = param_map["classes_path"]
        self.conf_thres = float(param_map["conf_thres"])
        self.iou_thres = float(param_map["iou_thres"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name_or_path"] = self.model_name_or_path
        param_map["model_name"] = self.model_name
        param_map["dataset"] = self.dataset
        param_map["model_path"] = self.model_path
        param_map["classes_path"] = self.classes_path
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["iou_thres"] = str(self.iou_thres)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class MaskRcnn(dataprocess.CInstanceSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CInstanceSegmentationTask.__init__(self, name)
        self.model = None
        self.class_names = []
        self.colors = []
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create parameters class
        if param is None:
            self.set_param_object(MaskRcnnParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def load_class_names(self):
        self.class_names.clear()
        param = self.get_param_object()

        with open(param.classes_path) as f:
            for row in f:
                self.class_names.append(row[:-1])

    def get_progress_steps(self):
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
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Get input :
        img_input = self.get_input(0)
        src_image = img_input.get_image()
        h, w, _ = src_image.shape

        # Step progress bar:
        self.emit_step_progress()

        # Load model
        if self.model is None or param.update:
            # Load class names
            self.load_class_names()

            if param.model_name_or_path != "":
                if os.path.isfile(param.model_name_or_path):
                    param.dataset = "Custom"
                    param.model_path = param.model_name_or_path
                else:
                    param.model_name = param.model_name_or_path

            # Load model
            use_torchvision = param.dataset != "Custom"
            self.model = models.mask_rcnn(use_pretrained=use_torchvision, classes=len(self.class_names))
            if param.dataset == "Custom":
                self.model.load_state_dict(torch.load(param.model_path, map_location=self.device))

            self.model.to(self.device)
            self.generate_colors()
            self.set_names(self.class_names)
            param.update = False

        pred = self.predict(src_image)
        cpu = torch.device("cpu")
        boxes = pred[0]["boxes"].to(cpu).numpy().tolist()
        scores = pred[0]["scores"].to(cpu).numpy().tolist()
        labels = pred[0]["labels"].to(cpu).numpy().tolist()
        masks = pred[0]["masks"]

        # Step progress bar:
        self.emit_step_progress()

        # Forward input image to result image
        self.forward_input_image(0, 0)

        # Get predictions
        valid_results = [scores.index(x) for x in scores if x > param.conf_thres]
        for i in valid_results:
            # box
            box_x = float(boxes[i][0])
            box_y = float(boxes[i][1])
            box_w = float(boxes[i][2] - boxes[i][0])
            box_h = float(boxes[i][3] - boxes[i][1])
            mask = (masks[i] > param.iou_thres).byte()
            self.add_object(i, 0, labels[i], float(scores[i]),
                                        box_x, box_y, box_w, box_h,
                                        mask.squeeze().cpu().numpy())

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class MaskRcnnFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_torchvision_mask_rcnn"
        self.info.short_description = "Mask R-CNN inference model for object detection and segmentation."
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
        self.info.documentation_link = "https://arxiv.org/abs/1703.06870"
        self.info.repository = "https://github.com/pytorch/vision"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.icon_path = "icons/pytorch-logo.png"
        self.info.version = "1.3.0"
        self.info.keywords = "torchvision,detection,segmentation,instance,object,resnet,pytorch"

    def create(self, param=None):
        # Create process object
        return MaskRcnn(self.info.name, param)
