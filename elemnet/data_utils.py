import numpy as np
import os
import re
import sys
from collections import defaultdict

import numpy
import pandas
import sklearn.preprocessing
#from matminer.featurizers.base import MultipleFeaturizer
from sklearn.model_selection import train_test_split

formulare = re.compile(r'([A-Z][a-z]*)(\d*\.*\d*)')

elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
            'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
            'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
            'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
            'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
            'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
            'Ds', 'Rg', 'Cn']


elements_tl = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K',
 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
 'Hg', 'Tl', 'Pb', 'Bi', 'Ac','Th', 'Pa', 'U', 'Np', 'Pu']

input1024 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '360', '361', '362', '363', '364', '365', '366', '367', '368', '369', '370', '371', '372', '373', '374', '375', '376', '377', '378', '379', '380', '381', '382', '383', '384', '385', '386', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396', '397', '398', '399', '400', '401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '417', '418', '419', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', '431', '432', '433', '434', '435', '436', '437', '438', '439', '440', '441', '442', '443', '444', '445', '446', '447', '448', '449', '450', '451', '452', '453', '454', '455', '456', '457', '458', '459', '460', '461', '462', '463', '464', '465', '466', '467', '468', '469', '470', '471', '472', '473', '474', '475', '476', '477', '478', '479', '480', '481', '482', '483', '484', '485', '486', '487', '488', '489', '490', '491', '492', '493', '494', '495', '496', '497', '498', '499', '500', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511', '512', '513', '514', '515', '516', '517', '518', '519', '520', '521', '522', '523', '524', '525', '526', '527', '528', '529', '530', '531', '532', '533', '534', '535', '536', '537', '538', '539', '540', '541', '542', '543', '544', '545', '546', '547', '548', '549', '550', '551', '552', '553', '554', '555', '556', '557', '558', '559', '560', '561', '562', '563', '564', '565', '566', '567', '568', '569', '570', '571', '572', '573', '574', '575', '576', '577', '578', '579', '580', '581', '582', '583', '584', '585', '586', '587', '588', '589', '590', '591', '592', '593', '594', '595', '596', '597', '598', '599', '600', '601', '602', '603', '604', '605', '606', '607', '608', '609', '610', '611', '612', '613', '614', '615', '616', '617', '618', '619', '620', '621', '622', '623', '624', '625', '626', '627', '628', '629', '630', '631', '632', '633', '634', '635', '636', '637', '638', '639', '640', '641', '642', '643', '644', '645', '646', '647', '648', '649', '650', '651', '652', '653', '654', '655', '656', '657', '658', '659', '660', '661', '662', '663', '664', '665', '666', '667', '668', '669', '670', '671', '672', '673', '674', '675', '676', '677', '678', '679', '680', '681', '682', '683', '684', '685', '686', '687', '688', '689', '690', '691', '692', '693', '694', '695', '696', '697', '698', '699', '700', '701', '702', '703', '704', '705', '706', '707', '708', '709', '710', '711', '712', '713', '714', '715', '716', '717', '718', '719', '720', '721', '722', '723', '724', '725', '726', '727', '728', '729', '730', '731', '732', '733', '734', '735', '736', '737', '738', '739', '740', '741', '742', '743', '744', '745', '746', '747', '748', '749', '750', '751', '752', '753', '754', '755', '756', '757', '758', '759', '760', '761', '762', '763', '764', '765', '766', '767', '768', '769', '770', '771', '772', '773', '774', '775', '776', '777', '778', '779', '780', '781', '782', '783', '784', '785', '786', '787', '788', '789', '790', '791', '792', '793', '794', '795', '796', '797', '798', '799', '800', '801', '802', '803', '804', '805', '806', '807', '808', '809', '810', '811', '812', '813', '814', '815', '816', '817', '818', '819', '820', '821', '822', '823', '824', '825', '826', '827', '828', '829', '830', '831', '832', '833', '834', '835', '836', '837', '838', '839', '840', '841', '842', '843', '844', '845', '846', '847', '848', '849', '850', '851', '852', '853', '854', '855', '856', '857', '858', '859', '860', '861', '862', '863', '864', '865', '866', '867', '868', '869', '870', '871', '872', '873', '874', '875', '876', '877', '878', '879', '880', '881', '882', '883', '884', '885', '886', '887', '888', '889', '890', '891', '892', '893', '894', '895', '896', '897', '898', '899', '900', '901', '902', '903', '904', '905', '906', '907', '908', '909', '910', '911', '912', '913', '914', '915', '916', '917', '918', '919', '920', '921', '922', '923', '924', '925', '926', '927', '928', '929', '930', '931', '932', '933', '934', '935', '936', '937', '938', '939', '940', '941', '942', '943', '944', '945', '946', '947', '948', '949', '950', '951', '952', '953', '954', '955', '956', '957', '958', '959', '960', '961', '962', '963', '964', '965', '966', '967', '968', '969', '970', '971', '972', '973', '974', '975', '976', '977', '978', '979', '980', '981', '982', '983', '984', '985', '986', '987', '988', '989', '990', '991', '992', '993', '994', '995', '996', '997', '998', '999', '1000', '1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010', '1011', '1012', '1013', '1014', '1015', '1016', '1017', '1018', '1019', '1020', '1021', '1022', '1023']

input512 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '360', '361', '362', '363', '364', '365', '366', '367', '368', '369', '370', '371', '372', '373', '374', '375', '376', '377', '378', '379', '380', '381', '382', '383', '384', '385', '386', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396', '397', '398', '399', '400', '401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '417', '418', '419', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', '431', '432', '433', '434', '435', '436', '437', '438', '439', '440', '441', '442', '443', '444', '445', '446', '447', '448', '449', '450', '451', '452', '453', '454', '455', '456', '457', '458', '459', '460', '461', '462', '463', '464', '465', '466', '467', '468', '469', '470', '471', '472', '473', '474', '475', '476', '477', '478', '479', '480', '481', '482', '483', '484', '485', '486', '487', '488', '489', '490', '491', '492', '493', '494', '495', '496', '497', '498', '499', '500', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511']

input256 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255']

input128 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127']

input64 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63']

input32 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']

phys_atts = ['0-norm', '2-norm', '3-norm', '5-norm', '7-norm', '10-norm', 'MagpieData minimum Number', 'MagpieData maximum Number', 'MagpieData range Number', 'MagpieData mean Number', 'MagpieData avg_dev Number', 'MagpieData mode Number', 'MagpieData minimum MendeleevNumber', 'MagpieData maximum MendeleevNumber', 'MagpieData range MendeleevNumber', 'MagpieData mean MendeleevNumber', 'MagpieData avg_dev MendeleevNumber', 'MagpieData mode MendeleevNumber', 'MagpieData minimum AtomicWeight', 'MagpieData maximum AtomicWeight', 'MagpieData range AtomicWeight', 'MagpieData mean AtomicWeight', 'MagpieData avg_dev AtomicWeight', 'MagpieData mode AtomicWeight', 'MagpieData minimum MeltingT', 'MagpieData maximum MeltingT', 'MagpieData range MeltingT', 'MagpieData mean MeltingT', 'MagpieData avg_dev MeltingT', 'MagpieData mode MeltingT', 'MagpieData minimum Column', 'MagpieData maximum Column', 'MagpieData range Column', 'MagpieData mean Column', 'MagpieData avg_dev Column', 'MagpieData mode Column', 'MagpieData minimum Row', 'MagpieData maximum Row', 'MagpieData range Row', 'MagpieData mean Row', 'MagpieData avg_dev Row', 'MagpieData mode Row', 'MagpieData minimum CovalentRadius', 'MagpieData maximum CovalentRadius', 'MagpieData range CovalentRadius', 'MagpieData mean CovalentRadius', 'MagpieData avg_dev CovalentRadius', 'MagpieData mode CovalentRadius', 'MagpieData minimum Electronegativity', 'MagpieData maximum Electronegativity', 'MagpieData range Electronegativity', 'MagpieData mean Electronegativity', 'MagpieData avg_dev Electronegativity', 'MagpieData mode Electronegativity', 'MagpieData minimum NsValence', 'MagpieData maximum NsValence', 'MagpieData range NsValence', 'MagpieData mean NsValence', 'MagpieData avg_dev NsValence', 'MagpieData mode NsValence', 'MagpieData minimum NpValence', 'MagpieData maximum NpValence', 'MagpieData range NpValence', 'MagpieData mean NpValence', 'MagpieData avg_dev NpValence', 'MagpieData mode NpValence', 'MagpieData minimum NdValence', 'MagpieData maximum NdValence', 'MagpieData range NdValence', 'MagpieData mean NdValence', 'MagpieData avg_dev NdValence', 'MagpieData mode NdValence', 'MagpieData minimum NfValence', 'MagpieData maximum NfValence', 'MagpieData range NfValence', 'MagpieData mean NfValence', 'MagpieData avg_dev NfValence', 'MagpieData mode NfValence', 'MagpieData minimum NValence', 'MagpieData maximum NValence', 'MagpieData range NValence', 'MagpieData mean NValence', 'MagpieData avg_dev NValence', 'MagpieData mode NValence', 'MagpieData minimum NsUnfilled', 'MagpieData maximum NsUnfilled', 'MagpieData range NsUnfilled', 'MagpieData mean NsUnfilled', 'MagpieData avg_dev NsUnfilled', 'MagpieData mode NsUnfilled', 'MagpieData minimum NpUnfilled', 'MagpieData maximum NpUnfilled', 'MagpieData range NpUnfilled', 'MagpieData mean NpUnfilled', 'MagpieData avg_dev NpUnfilled', 'MagpieData mode NpUnfilled', 'MagpieData minimum NdUnfilled', 'MagpieData maximum NdUnfilled', 'MagpieData range NdUnfilled', 'MagpieData mean NdUnfilled', 'MagpieData avg_dev NdUnfilled', 'MagpieData mode NdUnfilled', 'MagpieData minimum NfUnfilled', 'MagpieData maximum NfUnfilled', 'MagpieData range NfUnfilled', 'MagpieData mean NfUnfilled', 'MagpieData avg_dev NfUnfilled', 'MagpieData mode NfUnfilled', 'MagpieData minimum NUnfilled', 'MagpieData maximum NUnfilled', 'MagpieData range NUnfilled', 'MagpieData mean NUnfilled', 'MagpieData avg_dev NUnfilled', 'MagpieData mode NUnfilled', 'MagpieData minimum GSvolume_pa', 'MagpieData maximum GSvolume_pa', 'MagpieData range GSvolume_pa', 'MagpieData mean GSvolume_pa', 'MagpieData avg_dev GSvolume_pa', 'MagpieData mode GSvolume_pa', 'MagpieData minimum GSbandgap', 'MagpieData maximum GSbandgap', 'MagpieData range GSbandgap', 'MagpieData mean GSbandgap', 'MagpieData avg_dev GSbandgap', 'MagpieData mode GSbandgap', 'MagpieData minimum GSmagmom', 'MagpieData maximum GSmagmom', 'MagpieData range GSmagmom', 'MagpieData mean GSmagmom', 'MagpieData avg_dev GSmagmom', 'MagpieData mode GSmagmom', 'MagpieData minimum SpaceGroupNumber', 'MagpieData maximum SpaceGroupNumber', 'MagpieData range SpaceGroupNumber', 'MagpieData mean SpaceGroupNumber', 'MagpieData avg_dev SpaceGroupNumber', 'MagpieData mode SpaceGroupNumber', 'avg s valence electrons', 'avg p valence electrons', 'avg d valence electrons', 'avg f valence electrons', 'compound possible', 'max ionic char', 'avg ionic char']


input_atts = {'elements':elements, 'elements_tl':elements_tl, 'physical_atts': phys_atts, 'input1024':input1024, 'input512':input512, 'input256':input256, 'input128':input128, 'input64':input64, 'input32':input32}

elem_pos = dict()
i=0
for el in elements:
   elem_pos[el] = i
   i+=1


def parse_fractions(form):
    while '/' in form:
        di = form.index('/')
        num1 = [x for x in re.findall(r'\d*\.*\d*', form[:di]) if x != ''][-1]
        # print num1, 'x2 is:',x[di+1:]
        num2 = [x for x in re.findall(r'\d*\.*\d*', form[di + 1:]) if x != ''][0]
        # print x, 'num1:', num1, 'num2:', num2, 'xdi:', form[:di], 'xdi2:', form[di+1:]
        fract = '%.3f' % (float(num1) / float(num2))
        form = form[:di - len(num1)] + fract + form[di + len(num2) + 1:]
    return form

#parse_fractions('Mg1/3Ta2/3')

def parse_formula(formula):
    #weird_formula = {'SiC3c': 'SiC3C', '(VO)3(AsO4)2*2H2O': '(VO)3(AsO4)2H4O2', 'Zr1.05-1.08Se2': 'Zr1.05Se2.16',
    #                 'TI2O3': 'Ti2O3', 'SrTIO3': 'SrTiO3', 'SrTIO3': 'SrTiO3'}
    #if formula in weird_formula:
    #    formula = weird_formula[formula]
    # print 'working on ', formula
    pairs = formulare.findall(formula)
    stack = []
    curr_str = ''
    i = 0
    count_brack = 0
    res = defaultdict(int)
    formula = formula.replace('-', '').replace('@',
                                               '').replace(' ', '').replace('[', '(').replace(']', ')').replace('{',
                                                                                                                '(').replace(
        '}',
        ')').replace('@', '').replace('x', '').replace(' ', '')

    def parse_simple_formula(x):
        # print x,
        x = parse_fractions(x)
        # print x
        pairs = formulare.findall(x)
        length = sum((len(p[0]) + len(p[1]) for p in pairs))
        # print x,pairs, length, len(x)
        assert length == len(x)
        formula_dict = defaultdict(int)
        for el, sub in pairs:
            formula_dict[el] += float(sub) if sub else 1
        # print x, formula_dict
        return formula_dict

    while i < len(formula):
        # print 'curr:', formula[i], 'stac:', stack, res, ' form:', formula[i:]
        if formula[i] not in ['(', ')'] and not stack:
            curr_str = ''
            while i < len(formula) and formula[i] != '(':
                curr_str += formula[i]
                i += 1
            fract = re.findall(r'\d*\.*\d*', curr_str)[0]
            curr_str = curr_str[len(fract):]
            if not len(fract):
                fract = 1.
            else:
                fract = float(fract)
            temp_res = parse_simple_formula(curr_str)
            for k, v in temp_res.items():
                res[k] = temp_res[k] if k not in res else res[k] + temp_res[k]
        elif formula[i] not in [')']:
            stack.append(formula[i])
            i += 1
        else:
            i += 1
            fract = re.findall(r'\d*\.*\d*', formula[i:])[0]
            # print formula[i:], fract
            i = i + len(fract)
            if not len(fract):
                fract = 1.
            else:
                fract = float(fract)
            # print fract
            curr_str = ''
            while stack[-1] != '(':
                curr_str += stack.pop()
            stack.pop()
            curr_str = curr_str[::-1]
            fract1 = re.findall(r'\d*\.*\d*', curr_str)[0]
            if not len(fract1):
                fract *= 1.
            else:
                fract *= float(fract1)
            curr_str = curr_str[len(fract1):]
            temp_res = parse_simple_formula(curr_str)
            # print temp_res
            for k, v in temp_res.items():
                temp_res[k] *= fract
            # print 'updated:', temp_res
            if not stack:
                for k, v in temp_res.items():
                    res[k] = temp_res[k] if k not in res else res[k] + temp_res[k]
                    # res.update(temp_res)
            else:
                for i, v in temp_res.items():
                    stack.append(i)
                    stack.append(v)
    # print 'final:', formula, res
    if any([e for e in res if e in ['T', 'D', 'G', 'M', 'Q']]):
        print (formula, res)
    sum_nums = 1. * sum(res.values())
    for k in res: res[k] = 1. * res[k] / sum_nums
    return res

def get_fractions(comp):
    #print comp
    if all(e in elements_tl for e in comp):
        return np.array([comp[e] if e in comp else 0 for e in elements_tl], np.float32)
    else:   return None

def load_csv(train_data_path, val_data_path=None, test_data_path=None, input_types = None, label =None, test_size=None, val_size=0, logger=None):
    assert logger is not None
    logger.fprint('train data path is ', train_data_path)
    data_f = pandas.read_csv(train_data_path)
    logger.fprint('input attribute sets are: ', input_types)
    if test_data_path:
        logger.fprint('test data path is ', test_data_path)
        data_ft = pandas.read_csv(test_data_path)
    elif test_size:
        logger.fprint('splitting data into with test ratio=', test_size)
        data_f, data_ft = train_test_split(data_f, test_size=test_size, random_state=12345)
    else:
        data_ft = pd.DataFrame(columns=data_f.columns)

    if val_data_path:
        logger.fprint('val data path is ', val_data_path)
        data_fv = pandas.read_csv(val_data_path)
    elif val_size>0:
        data_fv = train_test_split(data_f, val_size=val_size, random_state=12345)
    else:
        data_fv= data_ft
    data_columns = data_f.columns
    if not input_types:
        input_attributes = data_columns[:-1]
        label = data_columns[-1]
    else:
        input_attributes = []
        for input_type in input_types:
            input_attributes += input_atts[input_type]
    logger.fprint('input attributes are: ', input_attributes)
    logger.fprint('label:', label)
    train_X = data_f[input_attributes].values
    train_y = data_f[label].values
    logger.fprint(data_f.describe())
    test_X = data_ft[input_attributes].values
    test_y = data_ft[label].values
    logger.fprint(data_ft.describe())
    valid_X = data_fv[input_attributes].values
    valid_y = data_fv[label].values
    logger.fprint(data_fv.describe())
    logger.fprint(' train, test, valid sizes: ', train_X.shape, train_y.shape, test_X.shape, test_y.shape, valid_X.shape, valid_y.shape)
    return train_X, train_y, valid_X, valid_y, test_X, test_y


#print get_fractions({'H':0.33, 'O':0.67})

#for comp in ['(3InAs)0.95(In2Te3)0.05', 'Mg1/3Ta2/3', 'KBr']:
#    print comp, parse_formula(comp)

