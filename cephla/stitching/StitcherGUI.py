# StitcherGUI.py
import sys
import os
from matplotlib import pyplot as plt
import gc
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLineEdit, QLabel, QProgressBar, QMessageBox, QCheckBox, QInputDialog, QComboBox, QSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import numpy as np
import napari
import psutil

from Stitcher import Stitcher

class StitchingThread(QThread):
    update_progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    warning = pyqtSignal(str)
    getting_flatfields = pyqtSignal()
    start_stitching = pyqtSignal()
    saving_started = pyqtSignal()
    saving_finished = pyqtSignal(str, object)


    def __init__(self, input_folder, output_name="output", output_format=".ome.zarr", apply_flatfield=0, use_registration=0, z_level=0, channel="", v_max_overlap=0, h_max_overlap=0):
        super().__init__()
        # Initialize Stitcher with the required attributes
        self.input_folder = input_folder
        self.output_format = output_format
        self.stitcher = Stitcher(input_folder=input_folder, output_name=output_name+output_format, apply_flatfield=apply_flatfield)
        self.output_path = self.stitcher.output_path
        self.use_registration = use_registration
        self.z_level = z_level
        self.channel = channel
        self.v_max_overlap = v_max_overlap
        self.h_max_overlap = h_max_overlap

    def run(self):
        try: 
            self.stitcher.parse_filenames()
            try:
                self.stitcher.extract_acquisition_parameters_from_json()
                self.stitcher.extract_selected_modes_from_xml()
                self.stitcher.determine_directions()
            except Exception as e:
                self.warning.emit(str(e))

            if self.stitcher.apply_flatfield:
                self.getting_flatfields.emit()
                self.stitcher.get_flatfields(progress_callback=self.update_progress.emit)

            self.start_stitching.emit()

            if self.use_registration: 
                # calculate shift from adjacent images and stitch with overlap
                vertical_shift, horizontal_shift = self.stitcher.calculate_shifts(self.z_level, self.channel, self.v_max_overlap, self.h_max_overlap)
                self.stitcher.pre_allocate_canvas(vertical_shift, horizontal_shift)
                self.stitcher.stitch_images_overlap(vertical_shift, horizontal_shift, progress_callback=self.update_progress.emit)
            else: 
                # stitch without overlap along a grid
                self.stitcher.pre_allocate_grid()
                self.stitcher.stitch_images(progress_callback=self.update_progress.emit)

            dz_um = self.stitcher.acquisition_params.get("dz(um)", None)
            sensor_pixel_size_um = self.stitcher.acquisition_params.get("sensor_pixel_size_um", None)
            self.saving_started.emit()
            if self.output_format == ".ome.tiff":
                self.stitcher.save_as_ome_tiff(dz_um=dz_um, sensor_pixel_size_um=sensor_pixel_size_um)
            elif self.output_format == ".ome.zarr":
                self.stitcher.save_as_ome_zarr(dz_um=dz_um, sensor_pixel_size_um=sensor_pixel_size_um)
            self.saving_finished.emit(self.stitcher.output_path, self.stitcher.dtype)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.stitcher.stitched_images = None
            self.stitcher.flatfields = {}
            self.finished.emit()

class StitchingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.output_path = ""
        self.output_format = ""
        self.dtype = None
        self.v_max_overlap = self.h_max_overlap = 0

    def initUI(self):
        self.layout = QVBoxLayout(self)
        
        # Input folder selection
        self.inputDirectoryBtn = QPushButton('Select Input Images Dataset Directory', self)
        self.inputDirectory = None
        self.inputDirectoryBtn.clicked.connect(self.selectInputDirectory)
        self.layout.addWidget(self.inputDirectoryBtn)

        # Checkbox for registering images when stitching
        self.useRegistrationCheck = QCheckBox('Align Image Edges When Stitching', self)
        self.useRegistrationCheck.toggled.connect(self.onRegistrationCheck)
        self.layout.addWidget(self.useRegistrationCheck)
        
        # Label to show selected max overlap after input dialog
        self.maxHorizontalOverlapLabel = QLabel('Select Max Overlap Between Horizontally Adjacent Images: ', self)
        self.layout.addWidget(self.maxHorizontalOverlapLabel)
        self.maxHorizontalOverlapLabel.hide()  # Hide initially

        self.maxVerticalOverlapLabel = QLabel('Select Max Overlap Between Vertically Adjacent Images: ', self)
        self.layout.addWidget(self.maxVerticalOverlapLabel)
        self.maxVerticalOverlapLabel.hide()  # Hide initially

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

        # Chekbox for applying flatfield correction
        self.applyFlatfieldCorrectionCheck = QCheckBox("Apply Flatfield Correction", self)
        self.layout.addWidget(self.applyFlatfieldCorrectionCheck)

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
        self.startBtn.clicked.connect(self.startProcess)
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
        prevInput = self.inputDirectory
        self.inputDirectory = QFileDialog.getExistingDirectory(self, "Select Input Image Folder")
        if self.inputDirectory: 
            self.inputDirectoryBtn.setText(f'Selected: {self.inputDirectory}')
            self.statusLabel.setText('Status: Input Images Folder Selected')
            if self.useRegistrationCheck.isChecked() and self.inputDirectory != prevInput:
                self.useRegistrationCheck.setChecked(False)

    def onRegistrationCheck(self, checked):
        if checked:
            if not self.inputDirectory:
                QMessageBox.warning(self, "Input Error", "Please Select an Input Image Folder First")
                self.useRegistrationCheck.setChecked(False)
                return
            stitcher = Stitcher(input_folder=self.inputDirectory)  # Temp instance to parse filenames
            stitcher.parse_filenames()
            h_max_overlap, h_ok = QInputDialog.getInt(self, "Max Horizontal Overlap",
                                "Enter Max Overlap for Horizontally Adjacent Tiles (pixels):",
                                128, 0, stitcher.input_width, 1)
            if h_ok:
                v_max_overlap, v_ok = QInputDialog.getInt(self, "Max Vertical Overlap", 
                                "Enter Max Overlap for Vertically Adjacent Tiles (pixels):", 
                                h_max_overlap, 0, stitcher.input_height, 1)
            if h_ok and v_ok:
                self.v_max_overlap, self.h_max_overlap = v_max_overlap, h_max_overlap
                self.maxHorizontalOverlapLabel.setText(f'Max Overlap for Horizontally Adjacent Images: {self.h_max_overlap} pixels')
                self.maxHorizontalOverlapLabel.show()
                self.maxVerticalOverlapLabel.setText(f'Max Overlap for Vertically Adjacent Images: {self.v_max_overlap} pixels')
                self.maxVerticalOverlapLabel.show()
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
            self.channelDropdown.clear()
            self.channelDropdown.addItems(stitcher.channel_names)
            self.channelDropdown.show()

        else:
            # Reset user inputs for registration
            self.v_max_overlap = self.h_max_overlap = 0
            self.maxHorizontalOverlapLabel.hide()
            self.maxVerticalOverlapLabel.hide()
            self.zLevelInputLabel.hide()
            self.zLevelInput.hide()
            self.channelDropdownLabel.hide()
            self.channelDropdown.clear()
            self.channelDropdown.hide()
            self.adjustSize()

    def startProcess(self):
        gc.collect()
        output_name = self.outputNameEdit.text().strip()
        if not self.inputDirectory or not output_name:
            QMessageBox.warning(self, "Input Error", "Please Select an Input Image Folder and Specify an Output Name.")
            return
        if self.outputFormatCombo.currentText() == "OME-TIFF":
            self.output_format = ".ome.tiff"
        elif self.outputFormatCombo.currentText() == "OME-ZARR":
            self.output_format = ".ome.zarr"

        temp_stitcher = Stitcher(self.inputDirectory, output_name)
        temp_stitcher.parse_filenames()  # Make sure this is necessary for the memory estimate
        estimated_memory = temp_stitcher.estimate_memory_usage()
        available_memory = psutil.virtual_memory().available

        # if estimated_memory > available_memory:
        #     msgBox = QMessageBox()
        #     msgBox.setIcon(QMessageBox.Warning)
        #     msgBox.setWindowTitle("Memory Warning")
        #     msgBox.setText("Insufficient memory for stitching operation.")
        #     msgBox.setInformativeText(f"Required: {estimated_memory} B\n"
        #                               f"Available: {available_memory} B\n")
        #     # Create a QLabel for the "Proceed anyway?" text and set alignment and styling if necessary
        #     proceed_label = QLabel("Proceed anyway?")
        #     proceed_label.setAlignment(Qt.AlignCenter)
        #     layout = msgBox.layout()
        #     layout.addWidget(proceed_label, 2, 0, 1, layout.columnCount(), Qt.AlignLeft)

        #     msgBox.addButton(QMessageBox.Yes)
        #     msgBox.addButton(QMessageBox.No)
        #     msgBox.setDefaultButton(QMessageBox.No)
        #     response = msgBox.exec()

        if estimated_memory > available_memory:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowTitle("Memory Warning")
            msgBox.setText("Proceed with Insufficient Memory for Image Stitching?")
            msgBox.setInformativeText(f"Required: {estimated_memory} B\n"
                                      f"Available: {available_memory} B\n")
            msgBox.addButton(QMessageBox.No)
            msgBox.addButton(QMessageBox.Yes)
            msgBox.setDefaultButton(QMessageBox.No)
            response = msgBox.exec()

            if response == QMessageBox.No:
                # Reset GUI or handle the abort operation
                #self.resetInputs()
                return  # Stop the process
        
        apply_flatfield = self.applyFlatfieldCorrectionCheck.isChecked()
        use_registration = self.useRegistrationCheck.isChecked()
        self.applyFlatfieldCorrectionCheck.setEnabled(False)
        self.useRegistrationCheck.setEnabled(False)
        if use_registration and (self.v_max_overlap < 0 or self.h_max_overlap < 0):
            QMessageBox.warning(self, "Input Error", "Please Enter Valid Max Overlap Values.")
            return

        selected_z_level = self.zLevelInput.value() if self.zLevelInput else 0
        selected_channel = self.channelDropdown.currentText() if self.channelDropdown else ""
        self.zLevelInput.hide()
        self.channelDropdown.hide()
        self.zLevelInputLabel.setText(f'Selected Z-Level for Registration: {selected_z_level}')
        self.channelDropdownLabel.setText(f'Selected Channel for Registration: {selected_channel}')

        self.viewNapariBtn.setEnabled(False)
        try:
            self.viewNapariBtn.clicked.disconnect()
        except TypeError:  # If no connections exist, disconnect will raise a TypeError
            pass
        self.progressBar.setValue(0)
        self.progressBar.show()
        self.statusLabel.setText('Status: Starting...')

        # Create and start the stitching thread
        self.thread = StitchingThread(self.inputDirectory, output_name, self.output_format, apply_flatfield, use_registration, selected_z_level, selected_channel, self.v_max_overlap, self.h_max_overlap)
        self.thread.update_progress.connect(self.updateProgressBar)
        self.thread.error_occurred.connect(self.showError)
        self.thread.warning.connect(self.showWarning)
        self.thread.getting_flatfields.connect(self.flatfieldStarted)
        self.thread.start_stitching.connect(self.startStitching)
        self.thread.saving_started.connect(self.savingStarted)
        self.thread.saving_finished.connect(self.savingFinished)
        self.thread.finished.connect(self.stitchingFinished)
        self.thread.start()

    def flatfieldStarted(self):
        self.statusLabel.setText('Status: Calculating Flatfield Images...')
        self.progressBar.setValue(0)

    def startStitching(self):
        self.progressBar.setValue(0)
        self.progressBar.show()
        self.statusLabel.setText('Status: Stitching Input Images...')

    def savingStarted(self):
        self.statusLabel.setText('Status: Saving Stitched Image...')
        self.progressBar.setRange(0, 0)  # Set the progress bar to indeterminate mode.

    def savingFinished(self, output_path, dtype):
        self.statusLabel.setText('Status: Saving Completed.')
        self.output_path = output_path
        self.dtype = dtype
        self.viewNapariBtn.setEnabled(True)
        self.contrast_limits = self.determineContrastLimits(dtype)
        try:
            self.viewNapariBtn.clicked.disconnect()
        except TypeError:  # If no connections exist, disconnect will raise a TypeError
            pass
        self.viewNapariBtn.clicked.connect(self.openNapari)

    def stitchingFinished(self):
        self.statusLabel.setText('Status: Done Stitching  ...' + os.path.join(*(self.output_path.split('/')[-3:])))
        # Reset the use registration checkbox
        self.useRegistrationCheck.setEnabled(True)
        self.useRegistrationCheck.setChecked(False)
        self.applyFlatfieldCorrectionCheck.setEnabled(True)
        self.applyFlatfieldCorrectionCheck.setChecked(False)
        self.maxHorizontalOverlapLabel.hide()
        self.maxVerticalOverlapLabel.hide()
        self.maxOverlap = None
        self.progressBar.hide()
        self.progressBar.setValue(0)
        self.progressBar.setRange(0, 100)
        self.adjustSize()

    def openNapari(self):
        try:
            viewer = napari.Viewer()
            if ".ome.zarr" in self.output_path:
                viewer.open(self.output_path, plugin='napari-ome-zarr', contrast_limits=self.contrast_limits)
            else:
                viewer.open(self.output_path, contrast_limits=self.contrast_limits)
            # Set contrast limits and colormap for each layer.

            colormap = 'inferno'  # 'blue', 'gray', 'magma', 'viridis', etc.
            for layer in viewer.layers:
                layer.colormap = colormap
                #layer.contrast_limits = self.contrast_limits

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

    def updateProgressBar(self, value, total):
        self.progressBar.setMaximum(total)
        self.progressBar.setValue(value)

    def showError(self, message):
        QMessageBox.critical(self, "Stitching Failed", message)
        self.statusLabel.setText('Status: Error Encountered!')
        self.progressBar.hide()

    def showWarning(self, message):
        warning = f"""Missing Acquisition Parameters, Configurations, or Coordinates: \n{message}
        \nContinuing Stitching..."""
        QMessageBox.warning(self, "File Not Found", warning)
        
def main():
    app = QApplication(sys.argv)
    ex = StitchingGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
