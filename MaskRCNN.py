from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class MaskRCNN(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from MaskRCNN.MaskRCNN_process import MaskRCNNProcessFactory
        # Instantiate process object
        return MaskRCNNProcessFactory()

    def getWidgetFactory(self):
        from MaskRCNN.MaskRCNN_widget import MaskRCNNWidgetFactory
        # Instantiate associated widget object
        return MaskRCNNWidgetFactory()
