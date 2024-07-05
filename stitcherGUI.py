import sys
import napari
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QProgressBar, QComboBox, QMessageBox, QCheckBox, QSpinBox, QLineEdit, QFileDialog)
from PyQt5.QtCore import QObject, pyqtSignal
from napari.utils.colormaps import Colormap, AVAILABLE_COLORMAPS
from stitcher import Stitcher, CHANNEL_COLORS_MAP

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
        self.outputNameLabel = QLabel('Enter Experiment Name', self)
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

        # Output path QLineEdit
        self.outputPathEdit = QLineEdit(self)
        self.outputPathEdit.setPlaceholderText("Enter Filepath To Visualize (No Stitching Required)")
        self.layout.addWidget(self.outputPathEdit)

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
            self.onRegistrationCheck(self.useRegistrationCheck.isChecked())

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
            self.outputPathEdit.setText(f"{self.inputDirectory}/{output_name}_complete_acquisition{output_format}")
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
            #self.stitcher.s_occurred.connect(self.onErrorOccurred)

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
        self.outputPathEdit.setText(path)
        self.output_path = path
        self.dtype = np.dtype(dtype)
        if dtype == np.uint16:
            c = [0, 65535]
        elif dtype == np.uint8:
            c = [0, 255]
        else:
            c = None
        self.contrast_limits = c
        self.setGeometry(300, 300, 500, 200)

    def onErrorOccurred(self, error):
        QMessageBox.critical(self, "Error", f"Error while processing: {error}")
        self.statusLabel.setText("Error occurred!")

    def onViewOutput(self):
        output_path = self.outputPathEdit.text()
        output_format = ".ome.zarr" if output_path.endswith(".ome.zarr") else None
        try:
            viewer = napari.Viewer()
            if ".ome.zarr" in output_path:
                viewer.open(output_path, plugin='napari-ome-zarr')
            else:
                viewer.open(output_path)

            for layer in viewer.layers:
                wavelength = self.extractWavelength(layer.name)
                channel_info = CHANNEL_COLORS_MAP.get(wavelength, {'hex': 0xFFFFFF, 'name': 'gray'})

                # Set colormap
                if channel_info['name'] in AVAILABLE_COLORMAPS:
                    layer.colormap = AVAILABLE_COLORMAPS[channel_info['name']]
                else:
                    layer.colormap = self.generateColormap(channel_info)

                # Set contrast limits based on dtype
                if np.issubdtype(layer.data.dtype, np.integer):
                    info = np.iinfo(layer.data.dtype)
                    layer.contrast_limits = (info.min, info.max)
                elif np.issubdtype(layer.data.dtype, np.floating):
                    layer.contrast_limits = (0.0, 1.0)

            napari.run()
        except Exception as e:
            QMessageBox.critical(self, "Error Opening in Napari", str(e))
            print(f"An error occurred while opening output in Napari: {e}")

    def extractWavelength(self, name):
        # Split the string and find the wavelength number immediately after "Fluorescence"
        parts = name.split()
        if 'Fluorescence' in parts:
            index = parts.index('Fluorescence') + 1
            if index < len(parts):
                return parts[index].split()[0]  # Assuming '488 nm Ex' and taking '488'
        for color in ['R', 'G', 'B']:
            if color in parts or "full_" + color in parts:
                return color
        return None

    def generateColormap(self, channel_info):
        """Convert a HEX value to a normalized RGB tuple."""
        c0 = (0, 0, 0)
        c1 = (((channel_info['hex'] >> 16) & 0xFF) / 255,  # Normalize the Red component
              ((channel_info['hex'] >> 8) & 0xFF) / 255,   # Normalize the Green component
              (channel_info['hex'] & 0xFF) / 255)          # Normalize the Blue component
        return Colormap(colors=[c0, c1], controls=[0, 1], name=channel_info['name'])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = StitchingGUI()
    gui.show()
    sys.exit(app.exec_())
