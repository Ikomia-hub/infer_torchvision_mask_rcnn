from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from infer_torchvision_mask_rcnn.infer_torchvision_mask_rcnn_process import MaskRcnnFactory
        # Instantiate process object
        return MaskRcnnFactory()

    def getWidgetFactory(self):
        from infer_torchvision_mask_rcnn.infer_torchvision_mask_rcnn_widget import MaskRcnnWidgetFactory
        # Instantiate associated widget object
        return MaskRcnnWidgetFactory()
