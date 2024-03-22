import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLineEdit, QLabel, QProgressBar, QMessageBox, QCheckBox, QInputDialog)
from PyQt5.QtCore import QThread, pyqtSignal

# Include the stitching functions from the script here or import them directly
from stitch import (extract_selected_modes_from_xml, extract_acquisition_parameters_from_json, parse_filenames, calculate_shifts, pre_allocate_arrays, pre_allocate_canvas, stitch_images, stitch_images_overlap, save_as_ome_tiff)

class StitchingThread(QThread):
    update_progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    saving_started = pyqtSignal()
    saving_finished = pyqtSignal()

    def __init__(self, input_folder, config_folder, output_name, use_registration, max_overlap, has_config_files):
        super().__init__()
        self.input_folder = input_folder
        self.output_name = output_name
        self.use_registration = use_registration
        self.has_config_files = has_config_files
        self.config_folder = config_folder
        self.max_overlap = max_overlap

    def run(self):
        try:
            # If there are configuration files, extract the selected modes and acquisition parameters
            if self.has_config_files:
                xml_file_path = os.path.join(self.config_folder, 'configurations.xml')
                json_file_path = os.path.join(self.config_folder, 'acquisition parameters.json')
                selected_modes = extract_selected_modes_from_xml(xml_file_path)
                acquisition_params = extract_acquisition_parameters_from_json(json_file_path)

            # Parse filenames to organize data
            channel_names, h, w, organized_data = parse_filenames(self.input_folder)

            # Pre-allocate arrays for stitching
            if self.use_registration:
                # Calculate shifts for stitching with registration
                v_shifts, h_shifts = calculate_shifts(self.input_folder, organized_data, self.max_overlap)  # max_overlap can be made dynamic
                stitched_images = pre_allocate_canvas(channel_names, h, w, organized_data, v_shifts, h_shifts)
                # Stitch images with overlap using registration
                stitched_images = stitch_images_overlap(self.input_folder, organized_data, stitched_images, channel_names, h, w, v_shifts, h_shifts, self.update_progress.emit)
            else:
                # Pre-allocate arrays without registration
                stitched_images = pre_allocate_arrays(channel_names, h, w, organized_data)
                # Stitch images without overlap
                stitched_images = stitch_images(self.input_folder, organized_data, stitched_images, channel_names, h, w, self.update_progress.emit)

            # Signal to indicate saving is starting
            self.saving_started.emit()

            # Save the stitched image with OME metadata
            if self.has_config_files:
                dz_um = acquisition_params.get("dz(um)", None)
                sensor_pixel_size_um = acquisition_params.get("sensor_pixel_size_um", None)
            else:
                dz_um = sensor_pixel_size_um = None  # Default values or some form of error handling

            output_folder_path = os.path.join(self.input_folder, "stitched")
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            output_path = os.path.join(output_folder_path, self.output_name + ".ome.tiff")
            save_as_ome_tiff(stitched_images, output_path, channel_names, dz_um, sensor_pixel_size_um)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.saving_finished.emit()
            self.finished.emit()

class StitchingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        
        # Input folder selection
        self.inputFolderBtn = QPushButton('Select Input Image Folder', self)
        self.inputFolderBtn.clicked.connect(self.selectInputFolder)
        self.layout.addWidget(self.inputFolderBtn)

        # Checkbox for selecting config files
        self.configCheckBox = QCheckBox("Provide Configurations and Acquisition Parameters", self)
        self.configCheckBox.toggled.connect(self.onConfigCheckBox)
        self.layout.addWidget(self.configCheckBox)

        # Checkbox for registering images when stitching
        self.useRegistrationCheck = QCheckBox('Align Image Edges When Stitching', self)
        self.useRegistrationCheck.toggled.connect(self.onRegistrationCheck)
        self.layout.addWidget(self.useRegistrationCheck)
        
        # Label to show selected max overlap after input dialog
        self.maxOverlapLabel = QLabel('Align Image Edges Selected: ', self)
        self.layout.addWidget(self.maxOverlapLabel)
        self.maxOverlapLabel.hide()  # Hide initially

        # Output name entry
        self.outputNameLabel = QLabel('Output Name (without .ome.tiff):', self)
        self.layout.addWidget(self.outputNameLabel)
        self.outputNameEdit = QLineEdit(self)
        self.layout.addWidget(self.outputNameEdit)

        # Start button
        self.startBtn = QPushButton('Start Stitching', self)
        self.startBtn.clicked.connect(self.startStitching)
        self.layout.addWidget(self.startBtn)

        # Progress bar
        self.progressBar = QProgressBar(self)
        self.layout.addWidget(self.progressBar)

        # Status label
        self.statusLabel = QLabel('Status: Idle', self)
        self.layout.addWidget(self.statusLabel)

        self.setWindowTitle('Image Stitcher')
        self.setGeometry(300, 300, 400, 200)

    def selectInputFolder(self):
        self.inputFolder = QFileDialog.getExistingDirectory(self, "Select Input Image Folder")
        if self.inputFolder: 
            self.inputFolderBtn.setText(f'Selected: {self.inputFolder}')
            self.statusLabel.setText('Status: Input images folder selected')

    def onConfigCheckBox(self, checked):
        if checked:
            self.configFolder = QFileDialog.getExistingDirectory(self, "Select Configuration Folder")
            if self.configFolder:
                self.configCheckBox.setText(f'Config Folder: {self.configFolder}')
                self.statusLabel.setText('Status: Configuration folder selected')
            else:
                # User canceled the selection, uncheck the checkbox
                self.configCheckBox.setChecked(False)
        else:
            self.configFolder = None

    def onRegistrationCheck(self, checked):
        if checked:
            max_overlap, ok = QInputDialog.getInt(self, "Max Overlap", "Enter Max Overlap Value:", 128, 0, 3000, 1)
            if ok:
                self.maxOverlap = max_overlap
                self.maxOverlapLabel.setText(f'Max Overlap Between Adjacent Images = {self.maxOverlap}')
                self.maxOverlapLabel.show()
            else:
                # User canceled the input dialog, uncheck the checkbox
                self.useRegistrationCheck.setChecked(False)
        else:
            self.maxOverlap = None
            self.maxOverlapLabel.hide()

    def startStitching(self):
        output_name = self.outputNameEdit.text().strip()
        use_registration = self.useRegistrationCheck.isChecked()
        has_config_files = self.configCheckBox.isChecked()

        if not self.inputFolder or not output_name:
            QMessageBox.warning(self, "Input Error", "Please select an input image folder and specify an output name.")
            return
        if use_registration and (not self.maxOverlap or self.maxOverlap < 0):
            QMessageBox.warning(self, "Input Error", "Please enter a valid max overlap value.")
            return

        self.progressBar.setValue(0)
        self.statusLabel.setText('Status: Starting stitching...')

        # Create and start the stitching thread
        self.thread = StitchingThread(self.inputFolder, self.configFolder, output_name, use_registration, self.maxOverlap, has_config_files)
        self.thread.update_progress.connect(self.updateProgressBar)
        self.thread.error_occurred.connect(self.showError)
        self.thread.saving_started.connect(self.savingStarted)
        self.thread.saving_finished.connect(self.savingFinished)
        self.thread.finished.connect(self.stitchingFinished)
        self.thread.start()

    def updateProgressBar(self, value, total):
        self.progressBar.setMaximum(total)
        self.progressBar.setValue(value)

    def stitchingFinished(self):
        self.statusLabel.setText('Status: Done stitching!')

    def showError(self, message):
        QMessageBox.critical(self, "Stitching Failed", message)
        self.statusLabel.setText('Status: Error encountered')

    def savingStarted(self):
        self.statusLabel.setText('Status: Saving stitched image...')

    def savingFinished(self):
        self.statusLabel.setText('Status: Saving completed.')

def main():
    app = QApplication(sys.argv)
    ex = StitchingGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()