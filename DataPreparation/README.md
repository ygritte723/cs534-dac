Creat_negs.ipynb:

This notebook was used to manually create negative samples for any cases where the original dac_create_negs process failed to produce them.

check_missing:

This script checks for any missing quality captions that do not have a corresponding image after using the dac_create_quality_caption tool.

collect_test_json:

Since dac_create_quality_caption generates one JSON file per image, this script combines all those individual JSON files into a single JSON file.

negs_and_pos:

This is derived from DAC/src/SVLC_learning. The negative and negative_auto classes were modified to return a list of negative captions. Additionally, initialization was added so that the list is created even if converting a positive caption to a negative one fails.

sample_test_images:

This script samples 1,000 test images from the vl_checklist.
