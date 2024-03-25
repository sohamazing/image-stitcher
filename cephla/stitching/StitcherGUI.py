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

    def __init__(self, input_folder, output_name, use_registration, max_overlap):
        super().__init__()
        # Initialize Stitcher with the required attributes
        self.input_folder = input_folder
        image_dir_path = os.path.join(self.input_folder, '0')
        if not os.path.isdir(image_dir_path):
            raise Exception(f"{input_folder}/0 is not a valid directory")
        self.stitcher = Stitcher(image_folder=image_dir_path, output_name=output_name, max_overlap=max_overlap)
        self.use_registration = use_registration

    def run(self):
        try:
            configs_path = os.path.join(self.input_folder, 'configurations.xml')
            acquistion_params_path = os.path.join(self.input_folder, 'acquisition parameters.json')
            self.stitcher.extract_selected_modes_from_xml(configs_path)
            self.stitcher.extract_acquisition_parameters_from_json(acquistion_params_path)
            self.stitcher.parse_filenames()

            if self.use_registration:
                v_shifts, h_shifts = self.stitcher.calculate_shifts_z()
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
        self.inputDirectoryBtn = QPushButton('Select Input Images Dataset Directory', self)
        self.inputDirectoryBtn.clicked.connect(self.selectInputDirectory)
        self.layout.addWidget(self.inputDirectoryBtn)


        # Checkbox for registering images when stitching
        self.useRegistrationCheck = QCheckBox('Align Image Edges When Stitching', self)
        self.useRegistrationCheck.toggled.connect(self.onRegistrationCheck)
        self.layout.addWidget(self.useRegistrationCheck)
        
        # Label to show selected max overlap after input dialog
        self.maxOverlapLabel = QLabel('Align Image Edges Selected: ', self)
        self.layout.addWidget(self.maxOverlapLabel)
        self.maxOverlap = None
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

    def selectInputDirectory(self):
        self.inputDirectory = QFileDialog.getExistingDirectory(self, "Select Input Image Folder")
        if self.inputDirectory: 
            self.inputDirectoryBtn.setText(f'Selected: {self.inputDirectory}')
            self.statusLabel.setText('Status: Input images folder selected')

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

        if not self.inputDirectory or not output_name:
            QMessageBox.warning(self, "Input Error", "Please select an input image folder and specify an output name.")
            return
        if use_registration and (not self.maxOverlap or self.maxOverlap < 0):
            QMessageBox.warning(self, "Input Error", "Please enter a valid max overlap value.")
            return

        self.progressBar.setValue(0)
        self.statusLabel.setText('Status: Starting stitching...')

        # Create and start the stitching thread
        self.thread = StitchingThread(self.inputDirectory, output_name, use_registration, self.maxOverlap)
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
        
        # Reset the use registration checkbox
        self.useRegistrationCheck.setChecked(False)
        self.maxOverlapLabel.hide()
        self.maxOverlap = None
        
        # Reset the max overlap label and value
        #self.maxOverlapLabel.setText('Max Overlap Between Adjacent Images: ')
        
        # Optionally, clear the input folder selection and output name
        #self.inputDirectoryBtn.setText('Select Input Image Folder')
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
