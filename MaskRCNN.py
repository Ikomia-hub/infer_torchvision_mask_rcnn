from ikomia import dataprocess
import MaskRCNN_process as processMod
import MaskRCNN_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class MaskRCNN(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.MaskRCNNProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.MaskRCNNWidgetFactory()
