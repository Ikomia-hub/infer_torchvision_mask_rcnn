import os
from ikomia import utils, core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_torchvision_mask_rcnn.infer_torchvision_mask_rcnn_process import MaskRcnnParam
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits core.CProtocolTaskWidget from Ikomia API
# --------------------
class MaskRcnnWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = MaskRcnnParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        self.combo_dataset = pyqtutils.append_combo(self.grid_layout, "Trained on")
        self.combo_dataset.addItem("Coco2017")
        self.combo_dataset.addItem("Custom")
        self.combo_dataset.setCurrentIndex(self._get_dataset_index())
        self.combo_dataset.currentIndexChanged.connect(self.on_combo_dataset_changed)

        self.browse_model = pyqtutils.append_browse_file(self.grid_layout, "Model path", self.parameters.model_path)

        self.browse_classes = pyqtutils.append_browse_file(self.grid_layout, "Classes path",
                                                           self.parameters.classes_path)

        self.spin_confidence = pyqtutils.append_double_spin(self.grid_layout, "Confidence",
                                                            self.parameters.conf_thres, 0.0, 1.0, 0.1, 2)
        self.spin_mask_thresh = pyqtutils.append_double_spin(self.grid_layout, "Mask threshold",
                                                             self.parameters.iou_thres, 0.0, 1.0, 0.1, 2)

        if self.parameters.dataset == "Coco2017":
            self.browse_model.set_path("Not used")
            self.browse_model.setEnabled(False)
            self.browse_classes.setEnabled(False)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def _get_dataset_index(self):
        if self.parameters.dataset == "Coco2017":
            return 0
        else:
            return 1

    def on_combo_dataset_changed(self, index):
        if self.combo_dataset.itemText(index) == "Coco2017":
            self.browse_model.set_path("Not used")
            self.browse_model.setEnabled(False)
            self.browse_classes.set_path(os.path.dirname(os.path.realpath(__file__)) + "/models/coco2017_classes.txt")
            self.browse_classes.setEnabled(False)
        else:
            self.browse_model.clear()
            self.browse_model.setEnabled(True)
            self.browse_classes.clear()
            self.browse_classes.setEnabled(True)

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.update = True
        self.parameters.dataset = self.combo_dataset.currentText()
        self.parameters.model_path = self.browse_model.path
        self.parameters.classes_path = self.browse_classes.path
        self.parameters.conf_thres = self.spin_confidence.value()
        self.parameters.iou_thres = self.spin_mask_thresh.value()

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits dataprocess.CWidgetFactory from Ikomia API
# --------------------
class MaskRcnnWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_torchvision_mask_rcnn"

    def create(self, param):
        # Create widget object
        return MaskRcnnWidget(param, None)
