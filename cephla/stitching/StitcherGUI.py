# StitcherGUI.py
import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLineEdit, QLabel, QProgressBar, QMessageBox, QCheckBox, QInputDialog)
from PyQt5.QtCore import QThread, pyqtSignal

from Stitcher import Stitcher

class StitchingThread(QThread):
    update_progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    saving_started = pyqtSignal()
    saving_finished = pyqtSignal()

    def __init__(self, input_folder, config_folder, output_name, use_registration, max_overlap, has_config_files):
        super().__init__()
        # Initialize Stitcher with the required attributes
        self.stitcher = Stitcher(input_folder=input_folder, output_name=output_name, max_overlap=max_overlap)
        self.config_folder = config_folder
        self.has_config_files = has_config_files
        self.use_registration = use_registration

    def run(self):
        try:
            if self.has_config_files:
                xml_file_path = os.path.join(self.config_folder, 'configurations.xml')
                json_file_path = os.path.join(self.config_folder, 'acquisition parameters.json')
                self.stitcher.extract_selected_modes_from_xml(xml_file_path)
                self.stitcher.extract_acquisition_parameters_from_json(json_file_path)
            
            self.stitcher.parse_filenames()

            if self.use_registration:
                v_shifts, h_shifts = self.stitcher.calculate_shifts()
                self.stitcher.pre_allocate_canvas(v_shifts, h_shifts)
                self.stitcher.stitch_images_overlap(v_shifts, h_shifts, progress_callback=self.update_progress.emit)
            else:
                self.stitcher.pre_allocate_arrays()
                self.stitcher.stitch_images(progress_callback=self.update_progress.emit)

            self.saving_started.emit()
            # Use parameters from self.stitcher.acquisition_params if available
            dz_um = self.stitcher.acquisition_params.get("dz(um)", None)
            sensor_pixel_size_um = self.stitcher.acquisition_params.get("sensor_pixel_size_um", None)
            self.stitcher.save_as_ome_tiff(dz_um=dz_um, sensor_pixel_size_um=sensor_pixel_size_um)

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
        self.configFolder = None
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
                self.configCheckBox.setText(f'Configurations and Acquisition Parameters Folder: {self.configFolder}')
                self.statusLabel.setText('Status: Acquisition parameters folder selected')
            else:
                # User canceled the selection, uncheck the checkbox
                self.configCheckBox.setChecked(False)
        else:
            self.configFolder = None

    def onRegistrationCheck(self, checked):
        if checked:
            max_overlap, ok = QInputDialog.getInt(self, "Max Overlap", "Enter Max Overlap (pixels):", 128, 0, 3000, 1)
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
        # Reset the config folder selection and checkbox
        self.configFolder = None
        self.configCheckBox.setChecked(False)
        self.configCheckBox.setText("Provide Configurations and Acquisition Parameters")
        
        # Reset the use registration checkbox
        self.useRegistrationCheck.setChecked(False)
        self.maxOverlapLabel.hide()
        
        # Reset the max overlap label and value
        self.maxOverlap = None
        self.maxOverlapLabel.setText('Max Overlap Between Adjacent Images: ')
        
        # Optionally, clear the input folder selection and output name
        #self.inputFolderBtn.setText('Select Input Image Folder')
        #self.outputNameEdit.clear()
        
        # You could also reset the progress bar here if you want
        self.progressBar.setValue(0)

    def showError(self, message):
        QMessageBox.critical(self, "Stitching Failed", message)
        self.statusLabel.setText('Status: Error encountered')

    def savingStarted(self):
        self.statusLabel.setText('Status: Saving stitched image...')
        self.progressBar.setRange(0, 0)  # Set the progress bar to indeterminate mode.


    def savingFinished(self):
        self.statusLabel.setText('Status: Saving completed.')
        self.progressBar.setRange(0, 100)  # Reset the progress bar to its normal range.
        self.progressBar.setValue(0)

def main():
    app = QApplication(sys.argv)
    ex = StitchingGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()