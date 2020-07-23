import unittest
from pathlib import Path

from synthxrd import xrdtools


class MetaDataTestCase(unittest.TestCase):
    def test_metadata(self):
        spec_lines = [
            "#L phi  H  K  L  Beta  Alpha  Epoch  io  i1  Bicron  AmpTek_sc  ROI1  ROI2  ROI3  IROI  imtot  immax  imroi2  imroi3  imroi4  imsca1  imsca2  imsca3  imsca4  transm  filters  corrdet  chiV1  chiV2  temperature  Seconds  imroi1",
            "132.0002 3.76332e-05 -0.0392959 0.117756 1.1690665 0.15936015 343.482 2620887 1783392 66838 111609 0 0 0 0 5.1133004e+08 12241.264 147399.95 -1945934.3 2.6613094e+08 12001.2 11888.17 12001.2 12241.264 1 0 2284608.5 0.076523995 0.075455863 25.08473 14.999859 34268805"
        ]
        scan = xrdtools.SpecScan(spec_lines)
        self.assertEqual(scan.metadata['temperature'].iloc[0], 25.08473)


class XMLTestCase(unittest.TestCase):
    def test_file_info(self):
        spec_lines = [
            '#UXML <group name="ad_file_info" NX_class="NXcollection" prefix="13PIL1MSi:">',
            '#UXML   <dataset name="file_format">TIFF</dataset>',
            '#UXML   <dataset name="file_path">/cars5/Data/gpd_user/data/bmc/2020/run2/fister/heater/images/s2_1/S002/</dataset>',
            '#UXML   <dataset name="file_name">s2_1_S002_00000</dataset>"',
            '#UXML   <dataset name="file_number" type="int">1</dataset>"',
            '#UXML   <dataset name="file_template">%s%s.tif</dataset>"',
            '#UXML   <dataset name="file_name_last_full">/cars5/Data/gpd_user/data/bmc/2020/run2/fister/heater/images/s2_1/S002/s2_1_S002_00000.tif</dataset>"',
            '#UXML </group>"',
            'An extra line',
        ]
        scan = xrdtools.SpecScan(spec_lines)
        self.assertEqual(str(scan.file_path), "s2_1/S002/s2_1_S002_00000.tif")

    def test_sequential_xml(self):
        """Make sure that having several xml entries in a row works okay."""
        spec_lines = [
            '#UXML <group name="ad_file_info" NX_class="NXcollection" prefix="13PIL1MSi:">',
            '#UXML   <dataset name="file_name_last_full">/cars5/Data/gpd_user/data/bmc/2020/run2/fister/heater/images/s2_1/S002/s2_1_S002_00000.tif</dataset>"',
            '#UXML </group>"',
            '#UXML <group name="ad_detector" NX_class="NXdetector" prefix="13PIL1MSi:" array_port="PIL">',
            '#UXML </group>"',
        ]
        scan = xrdtools.SpecScan(spec_lines)
        self.assertEqual(str(scan.file_path), "s2_1/S002/s2_1_S002_00000.tif")
        

class FullSpecTestCase(unittest.TestCase):
    def test_sample_spec_file(self):
        samples = xrdtools.parse_spec_file(Path('./sample_spec_file.spec'))
        print(samples)


class ImportXrdTestCase(unittest.TestCase):
    def test_mismatched_sample_names(self):
        with self.assertRaises(ValueError):
            xrdtools.import_xrd(Path('./sample_spec_file.spec'),
                                     sample_names={})
