# stitcherGUI_V2.py
import sys
import napari
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QProgressBar, QComboBox, QMessageBox, QCheckBox, QSpinBox, QLineEdit, QFileDialog)
from PyQt5.QtCore import QObject, pyqtSignal

from stitcher_V2 import Stitcher  # Make sure to import your actual stitcher class

class StitchingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.stitcher = None  # Stitcher is initialized when needed
        self.inputDirectory = None  # This will be set by the directory selection
        self.output_path = ""
        self.dtype = None
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        # Input Directory Selection
        self.inputDirectoryBtn = QPushButton('Select Input Directory', self)
        self.inputDirectoryBtn.clicked.connect(self.selectInputDirectory)
        self.layout.addWidget(self.inputDirectoryBtn)

        # Checkbox for applying flatfield correction
        self.applyFlatfieldCheck = QCheckBox("Apply Flatfield Correction", self)
        self.layout.addWidget(self.applyFlatfieldCheck)

        # Checkbox for enabling registration
        self.useRegistrationCheck = QCheckBox('Use Registration', self)
        self.useRegistrationCheck.toggled.connect(self.onRegistrationCheck)
        self.layout.addWidget(self.useRegistrationCheck)

        # Spin box for selecting Z-Level if registration is used
        self.zLevelLabel = QLabel('Enter Registration Z-Level', self)
        self.zLevelInput = QSpinBox(self)
        self.zLevelInput.setMinimum(0)
        self.zLevelInput.setMaximum(100)  # Adjust max as necessary based on your application
        self.zLevelLabel.hide()
        self.zLevelInput.hide()  # Initially hidden
        self.layout.addWidget(self.zLevelLabel)
        self.layout.addWidget(self.zLevelInput)

        # Combo box for selecting channel for registration if used
        self.channelLabel = QLabel('Enter Registration Channel', self)
        self.channelCombo = QComboBox(self)
        self.channelLabel.hide()
        self.channelCombo.hide()  # Initially hidden
        self.layout.addWidget(self.channelLabel)
        self.layout.addWidget(self.channelCombo)

        # Output format combo box with OME-ZARR as the first option
        self.outputFormatCombo = QComboBox()
        self.outputFormatCombo.addItems(['OME-ZARR', 'OME-TIFF'])
        self.layout.addWidget(self.outputFormatCombo)

        # Output name entry
        self.outputNameLabel = QLabel('Enter Output Name (no extension)', self)

        self.outputNameEdit = QLineEdit(self)
        self.layout.addWidget(self.outputNameLabel)
        self.layout.addWidget(self.outputNameEdit)

        # Progress bar setup
        self.progressBar = QProgressBar(self)
        self.progressBar.hide()
        self.layout.addWidget(self.progressBar)

        # Status label
        self.statusLabel = QLabel('Status: Ready', self)
        self.layout.addWidget(self.statusLabel)

        # Start stitching button
        self.startBtn = QPushButton('Start Stitching', self)
        self.startBtn.clicked.connect(self.onStitchingStart)
        self.layout.addWidget(self.startBtn)

        # View in Napari button
        self.viewBtn = QPushButton('View Output in Napari', self)
        self.viewBtn.clicked.connect(self.onViewOutput)
        self.viewBtn.setEnabled(False)
        self.layout.addWidget(self.viewBtn)

        self.setWindowTitle('Cephla Image Stitcher')
        self.setGeometry(300, 300, 500, 200)
        self.show()

    def selectInputDirectory(self):
        self.inputDirectory = QFileDialog.getExistingDirectory(self, "Select Input Image Folder")
        if self.inputDirectory:
            self.inputDirectoryBtn.setText(f'Selected: {self.inputDirectory}')

    def onRegistrationCheck(self, checked):
        if checked:
            if not self.inputDirectory:
                QMessageBox.warning(self, "Input Error", "Please Select an Input Image Folder First")
                self.useRegistrationCheck.setChecked(False)
                return

            try:
                # Create temporary Stitcher to parse filenames
                stitcher = Stitcher(input_folder=self.inputDirectory)
                timepoints = stitcher.get_time_points(input_folder=self.inputDirectory)
                
                if not timepoints:
                    QMessageBox.warning(self, "Input Error", "No time points found in the selected directory.")
                    self.useRegistrationCheck.setChecked(False)
                    return

                stitcher.parse_filenames(time_point=timepoints[0])

                # Setup Z-Level
                self.zLevelLabel.show()
                self.zLevelInput.setMinimum(0)
                self.zLevelInput.setMaximum(stitcher.num_z - 1)
                self.zLevelInput.show()

                # Setup channel dropdown
                self.channelLabel.show()
                self.channelCombo.clear()
                self.channelCombo.addItems(stitcher.channel_names)
                self.channelCombo.show()
            except Exception as e:
                QMessageBox.critical(self, "Parsing Error", f"An error occurred during data processing: {e}")
                self.useRegistrationCheck.setChecked(False)
                self.zLevelLabel.hide()
                self.zLevelInput.hide()
                self.channelLabel.hide()
                self.channelCombo.hide()
        else:
            self.zLevelLabel.hide()
            self.zLevelInput.hide()
            self.channelLabel.hide()
            self.channelCombo.hide()
        

    def onStitchingStart(self):
        self.statusLabel.setText('Status: Stitching...')
        output_name = self.outputNameEdit.text().strip()
        output_format = '.' + self.outputFormatCombo.currentText().lower().replace('-', '.')
        use_registration = self.useRegistrationCheck.isChecked()
        apply_flatfield = self.applyFlatfieldCheck.isChecked()

        if not self.inputDirectory:
            QMessageBox.warning(self, "Input Error", "Please select an input directory.")
            return
        if not output_name:
            QMessageBox.warning(self, "Input Error", "Please enter an output name.")
            return
        
        if use_registration:
            # Assuming z-level and channel are required for registration
            z_level = self.zLevelInput.value()
            channel = self.channelCombo.currentText()
            if not channel:  # Add check to ensure a channel is selected
                QMessageBox.warning(self, "Input Error", "Please select a registration channel.")
                return
        else:
            z_level = 0
            channel = ''

        try:
            self.stitcher = Stitcher(
                input_folder=self.inputDirectory,
                output_name=output_name,
                output_format=output_format,
                apply_flatfield=apply_flatfield,
                use_registration=use_registration,
                registration_channel=channel,
                registration_z_level=z_level,
            )
            self.setupConnections()
            self.stitcher.start()
            self.progressBar.show()
        except Exception as e:
            QMessageBox.critical(self, "Stitching Error", str(e))
            self.statusLabel.setText('Status: Error Encountered')

    def setupConnections(self):
        if self.stitcher:
            self.stitcher.update_progress.connect(self.updateProgressBar)
            self.stitcher.getting_flatfields.connect(lambda: self.statusLabel.setText("Calculating flatfields..."))
            self.stitcher.starting_stitching.connect(lambda: self.statusLabel.setText("Stitching images..."))
            self.stitcher.starting_saving.connect(self.onStartingSaving)
            self.stitcher.finished_saving.connect(self.onFinishedSaving)
            self.stitcher.error_occurred.connect(self.onErrorOccurred)

    def updateProgressBar(self, value, maximum):
        self.progressBar.setRange(0, maximum)
        self.progressBar.setValue(value)

    def onStartingSaving(self, stitch_complete=False):
        if stitch_complete:
            self.statusLabel.setText('Status: Saving Complete Acquisition Image...')
        else:
            self.statusLabel.setText('Status: Saving Stitched Image...')
        self.progressBar.setRange(0, 0)  # Indeterminate mode
        self.progressBar.show()
        self.statusLabel.show()

    def onFinishedSaving(self, path, dtype):
        self.progressBar.setValue(0)
        self.progressBar.hide()
        self.viewBtn.setEnabled(True)
        self.statusLabel.setText("Saving completed. Ready to view.")
        self.output_path = path
        self.dtype = np.dtype(dtype)
        self.contrast_limits = self.determineContrastLimits(self.dtype)
        self.setGeometry(300, 300, 500, 200)

    def determineContrastLimits(self, dtype):
        if dtype == np.uint16:
            return [0, 65535]
        elif dtype == np.uint8:
            return [0, 255]
        return None

    def onErrorOccurred(self, error):
        QMessageBox.critical(self, "Error", f"Error while processing: {error}")
        self.statusLabel.setText("Error occurred!")

    def onViewOutput(self):
        try:
            viewer = napari.Viewer()
            viewer.open(self.output_path, plugin='napari-ome-zarr', contrast_limits=self.contrast_limits)
            napari.run(max_loop_level=2)
        except Exception as e:
            QMessageBox.critical(self, "Error Opening in Napari", str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = StitchingGUI()
    gui.show()
    sys.exit(app.exec_())
