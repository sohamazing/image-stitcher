# StitcherGUI.py
import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLineEdit, QLabel, QProgressBar, QMessageBox, QCheckBox, QInputDialog, QComboBox, QSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import napari

from Stitcher import Stitcher

class StitchingThread(QThread):
    update_progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    warning = pyqtSignal(str)
    saving_started = pyqtSignal()
    saving_finished = pyqtSignal(str, object)


    def __init__(self, input_folder, output_name="output", output_format=".ome.zarr", use_registration=0, max_overlap=0, z_level=0, channel=""):
        super().__init__()
        # Initialize Stitcher with the required attributes
        self.input_folder = input_folder
        self.output_format = output_format
        self.stitcher = Stitcher(input_folder=input_folder, output_name=output_name+output_format)
        self.output_path = self.stitcher.output_path
        self.use_registration = use_registration
        self.z_level = z_level
        self.channel = channel
        self.max_overlap = max_overlap

    def run(self):
        try:
            # get acquisition parameters
            configs_path = os.path.join(self.input_folder, 'configurations.xml')
            acquistion_params_path = os.path.join(self.input_folder, 'acquisition parameters.json')
            try:
                self.stitcher.extract_acquisition_parameters_from_json(acquistion_params_path)
                #self.stitcher.extract_selected_modes_from_xml(configs_path)
            except Exception as e:
                self.warning.emit(str(e))
            # parse filenames in input directory
            self.stitcher.parse_filenames()
            if self.use_registration: 
                # calculate shift from adjacent images and stitch with overlap
                vertical_shift, horizontal_shift = self.stitcher.calculate_shifts(self.z_level, self.channel, self.max_overlap)
                self.stitcher.pre_allocate_canvas(vertical_shift, horizontal_shift)
                self.stitcher.stitch_images_overlap(vertical_shift, horizontal_shift, progress_callback=self.update_progress.emit)
            else: 
                # stitch without overlap along a grid
                self.stitcher.pre_allocate_grid()
                self.stitcher.stitch_images(progress_callback=self.update_progress.emit)

            self.saving_started.emit()
            # use acquisition parameters if available
            dz_um = self.stitcher.acquisition_params.get("dz(um)", None)
            sensor_pixel_size_um = self.stitcher.acquisition_params.get("sensor_pixel_size_um", None)

            # save output with ome metadata
            if self.output_format == ".ome.tiff":
                self.stitcher.save_as_ome_tiff(dz_um=dz_um, sensor_pixel_size_um=sensor_pixel_size_um)
            elif self.output_format == ".ome.zarr":
                self.stitcher.save_as_ome_zarr(dz_um=dz_um, sensor_pixel_size_um=sensor_pixel_size_um)
            self.saving_finished.emit(self.stitcher.output_path, self.stitcher.dtype)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.finished.emit()

class StitchingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.output_path = ""
        self.output_format = ""
        self.dtype = None
        self.max_overlap = 0

    def initUI(self):
        self.layout = QVBoxLayout(self)
        
        # Input folder selection
        self.inputDirectoryBtn = QPushButton('Select Input Images Dataset Directory', self)
        self.inputDirectoryBtn.clicked.connect(self.selectInputDirectory)
        self.inputDirectory = None
        self.layout.addWidget(self.inputDirectoryBtn)


        # Checkbox for registering images when stitching
        self.useRegistrationCheck = QCheckBox('Align Image Edges When Stitching', self)
        self.useRegistrationCheck.toggled.connect(self.onRegistrationCheck)
        self.layout.addWidget(self.useRegistrationCheck)
        
        # Label to show selected max overlap after input dialog
        self.maxOverlapLabel = QLabel('Select Max Overlap Between Adjacent Images: ', self)
        self.layout.addWidget(self.maxOverlapLabel)
        self.maxOverlapLabel.hide()  # Hide initially

        # Label to show selected z-Level 
        self.zLevelInputLabel = QLabel('Select Z-Level for Registration: ', self)
        self.layout.addWidget(self.zLevelInputLabel)
        self.zLevelInputLabel.hide()
        self.zLevelInput = QSpinBox(self)
        self.layout.addWidget(self.zLevelInput)
        self.zLevelInput.hide()

        # Label to show selected channel
        self.channelDropdownLabel = QLabel('Select Channel for Registration: ', self)
        self.layout.addWidget(self.channelDropdownLabel)
        self.channelDropdownLabel.hide()
        self.channelDropdown = QComboBox(self)
        self.layout.addWidget(self.channelDropdown)
        self.channelDropdown.hide()

        # Output format selection
        self.outputFormatLabel = QLabel('Select Output Format:', self)
        self.layout.addWidget(self.outputFormatLabel)
        self.outputFormatCombo = QComboBox(self)
        self.outputFormatCombo.addItem("OME-ZARR")
        self.outputFormatCombo.addItem("OME-TIFF")
        self.layout.addWidget(self.outputFormatCombo)

        # Output name entry
        self.outputNameLabel = QLabel('Output Name (without extension):', self)
        self.layout.addWidget(self.outputNameLabel)
        self.outputNameEdit = QLineEdit(self)
        self.layout.addWidget(self.outputNameEdit)

        # Start button
        self.startBtn = QPushButton('Start Stitching', self)
        self.startBtn.clicked.connect(self.startStitching)
        self.layout.addWidget(self.startBtn)

        # View napari button
        self.viewNapariBtn = QPushButton('View Output in Napari', self)
        self.viewNapariBtn.setEnabled(False)  # Initially disabled until saving is finished
        self.layout.addWidget(self.viewNapariBtn)

        # Progress bar
        self.progressBar = QProgressBar(self)
        self.layout.addWidget(self.progressBar)
        self.progressBar.hide()

        # Status label
        self.statusLabel = QLabel('Status: Enter Image Stitcher Inputs', self)
        self.layout.addWidget(self.statusLabel)

        # Window title
        self.setWindowTitle('Image Stitcher')
        self.setGeometry(300, 300, 400, 200)

    def selectInputDirectory(self):
        self.inputDirectory = QFileDialog.getExistingDirectory(self, "Select Input Image Folder")
        if self.inputDirectory: 
            self.inputDirectoryBtn.setText(f'Selected: {self.inputDirectory}')
            self.statusLabel.setText('Status: Input Images Folder Selected')

    def onRegistrationCheck(self, checked):
        if checked:
            if not self.inputDirectory:
                QMessageBox.warning(self, "Input Error", "Please Select an Input Image Folder First")
                self.useRegistrationCheck.setChecked(False)
                return
            stitcher = Stitcher(input_folder=self.inputDirectory)  # Temp instance to parse filenames
            stitcher.parse_filenames()
            max_max_overlap = min(stitcher.input_height, stitcher.input_width)
            max_overlap, ok = QInputDialog.getInt(self, "Max Overlap", "Enter Max Overlap (pixels):", 128, 0, max_max_overlap, 1)
            if ok:
                self.max_overlap = max_overlap
                self.maxOverlapLabel.setText(f'Max Overlap Between Adjacent Images: {self.max_overlap} pixels')
                self.maxOverlapLabel.show()
            else:
                # User canceled the input dialog, uncheck the checkbox
                self.useRegistrationCheck.setChecked(False)
                return

            # Create z-level input
            self.zLevelInputLabel.setText('Select Z-Level for Registration:')
            self.zLevelInputLabel.show()
            self.zLevelInput.setMinimum(0)  # Minimum z level
            self.zLevelInput.setMaximum(stitcher.num_z - 1)  # Maximum z level
            self.zLevelInput.show()

            # Create channel dropdown
            self.channelDropdownLabel.setText('Select Channel for Registration:')
            self.channelDropdownLabel.show()
            self.channelDropdown.addItems(stitcher.channel_names)
            self.channelDropdown.show()

        else:
            # Reset user inputs for registration
            self.max_overlap = 0
            self.maxOverlapLabel.hide()
            self.zLevelInputLabel.hide()
            self.zLevelInput.hide()
            self.channelDropdownLabel.hide()
            self.channelDropdown.clear()
            self.channelDropdown.hide()
            self.adjustSize()

    def startStitching(self):
        output_name = self.outputNameEdit.text().strip()
        if self.outputFormatCombo.currentText() == "OME-TIFF":
            self.output_format = ".ome.tiff"
        elif self.outputFormatCombo.currentText() == "OME-ZARR":
            self.output_format = ".ome.zarr"
        
        use_registration = self.useRegistrationCheck.isChecked()
        self.useRegistrationCheck.setEnabled(False)

        selected_z_level = self.zLevelInput.value() if self.zLevelInput else 0
        self.zLevelInputLabel.setText(f'Selected Z-Level for Registration: {selected_z_level}')
        self.zLevelInput.hide()
        
        selected_channel = self.channelDropdown.currentText() if self.channelDropdown else ""
        self.channelDropdownLabel.setText(f'Selected Channel for Registration: {selected_channel}')
        self.channelDropdown.hide()

        if not self.inputDirectory or not output_name:
            QMessageBox.warning(self, "Input Error", "Please Select an Input Image Folder and Specify an Output Name.")
            return
        if use_registration and (not self.max_overlap or self.max_overlap < 0):
            QMessageBox.warning(self, "Input Error", "Please Enter a Valid Max Overlap Value.")
            return

        self.viewNapariBtn.setEnabled(False)
        self.progressBar.setValue(0)
        self.progressBar.show()
        self.statusLabel.setText('Status: Starting Stitching...')

        # Create and start the stitching thread
        self.thread = StitchingThread(self.inputDirectory, output_name, self.output_format, use_registration, self.max_overlap, selected_z_level, selected_channel)
        self.thread.update_progress.connect(self.updateProgressBar)
        self.thread.error_occurred.connect(self.showError)
        self.thread.warning.connect(self.showWarning)
        self.thread.saving_started.connect(self.savingStarted)
        self.thread.saving_finished.connect(self.savingFinished)
        self.thread.finished.connect(self.stitchingFinished)
        self.thread.start()

    def updateProgressBar(self, value, total):
        self.progressBar.setMaximum(total)
        self.progressBar.setValue(value)

    def stitchingFinished(self):
        self.statusLabel.setText('Status: Done Stitching!')
        # Reset the use registration checkbox
        self.useRegistrationCheck.setEnabled(True)
        self.useRegistrationCheck.setChecked(False)
        self.maxOverlapLabel.hide()
        self.maxOverlap = None
        
        # You could also reset the progress bar here if you want
        self.progressBar.setValue(0)

    def showError(self, message):
        QMessageBox.critical(self, "Stitching Failed", message)
        self.statusLabel.setText('Status: Error Encountered!')
        self.progressBar.hide()

    def showWarning(self, message):
        QMessageBox.warning(self, "File Not Found Warning:", message)
        self.statusLabel.setText('Status: File Not Found... Continuing Stitching...')

    def savingStarted(self):
        self.statusLabel.setText('Status: Saving Stitched Image...')
        self.progressBar.setRange(0, 0)  # Set the progress bar to indeterminate mode.

    def savingFinished(self, output_path, dtype):
        self.statusLabel.setText('Status: Saving Completed.')
        self.adjustSize()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.hide()
        self.output_path = output_path
        self.dtype = dtype
        self.viewNapariBtn.setEnabled(True)
        self.contrast_limits = self.determineContrastLimits(dtype)

        try:
            self.viewNapariBtn.clicked.disconnect()
        except TypeError:  # If no connections exist, disconnect will raise a TypeError
            pass
        self.viewNapariBtn.clicked.connect(self.openNapari)

    def openNapari(self):
        try:
            viewer = napari.Viewer()
            if ".ome.zarr" in self.output_path:
                viewer.open(self.output_path, plugin='napari-ome-zarr', contrast_limits=self.contrast_limits)
            else:
                viewer.open(self.output_path, contrast_limits=self.contrast_limits)
            '''
            # Apply contrast limits if defined
             if contrast_limits is not None:
                for layer in viewer.layers:
                    layer.contrast_limits = contrast_limits\
            '''
            # Start the Napari event loop
            # napari.run()
        except Exception as e:
            QMessageBox.critical(self, "Error Opening in Napari", str(e))

    def determineContrastLimits(self, dtype):
        if dtype == np.uint16:
            return [0, 65535]
        elif dtype == np.uint8:
            return [0, 255]
        return None
                   
        
def main():
    app = QApplication(sys.argv)
    ex = StitchingGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
