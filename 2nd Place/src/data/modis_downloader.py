import os
import wget
from pathlib import Path

from azure.storage.blob import ContainerClient

temporalr = [str(i) for i in range(2022001, 2022182)]

temporalr_archive = ['2014133', '2014007', '2014140', '2014014', '2014147', '2014021', '2014154',
                     '2014028', '2014161', '2014035', '2014168', '2014042', '2014175', '2014049',
                     '2014056', '2014063', '2014070', '2014077', '2014336', '2014084', '2014343',
                     '2014091', '2014350', '2014098', '2014357', '2014105', '2014364', '2014112',
                     '2014119', '2014126', '2015132', '2015006', '2015139', '2015013', '2015146',
                     '2015020', '2015153', '2015027', '2015160', '2015034', '2015167', '2015041',
                     '2015174', '2015048', '2015181', '2015055', '2015062', '2015069', '2015076',
                     '2015335', '2015083', '2015342', '2015090', '2015349', '2015097', '2015356',
                     '2015104', '2015363', '2015111', '2015118', '2015125', '2016130', '2016131',
                     '2016005', '2016138', '2016012', '2016145', '2016019', '2016148', '2016152',
                     '2016026', '2016159', '2016033', '2016166', '2016039', '2016040', '2016173',
                     '2016047', '2016178', '2016180', '2016054', '2016061', '2016068', '2016075',
                     '2016082', '2016341', '2016086', '2016089', '2016092', '2016348', '2016094',
                     '2016095', '2016096', '2016098', '2016355', '2016103', '2016362', '2016107',
                     '2016110', '2016117', '2016124', '2017129', '2017003', '2017136', '2017010',
                     '2017143', '2017017', '2017150', '2017024', '2017028', '2017029', '2017157',
                     '2017031', '2017164', '2017038', '2017171', '2017045', '2017178', '2017052',
                     '2017059', '2017066', '2017073', '2017080', '2017339', '2017087', '2017346',
                     '2017094', '2017353', '2017101', '2017360', '2017108', '2017115', '2017122',
                     '2018128', '2018002', '2018135', '2018009', '2018142', '2018016', '2018144',
                     '2018148', '2018149', '2018023', '2018152', '2018153', '2018156', '2018030',
                     '2018163', '2018037', '2018170', '2018044', '2018177', '2018051', '2018058',
                     '2018063', '2018065', '2018072', '2018079', '2018338', '2018086', '2018089',
                     '2018090', '2018345', '2018093', '2018352', '2018100', '2018359', '2018107',
                     '2018112', '2018113', '2018114', '2018115', '2018116', '2018121', '2019001',
                     '2019134', '2019008', '2019141', '2019015', '2019148', '2019022', '2019155',
                     '2019156', '2019029', '2019159', '2019160', '2019161', '2019162', '2019036',
                     '2019164', '2019165', '2019169', '2019043', '2019175', '2019176', '2019050',
                     '2019057', '2019064', '2019068', '2019071', '2019074', '2019075', '2019076',
                     '2019078', '2019337', '2019083', '2019084', '2019085', '2019088', '2019344',
                     '2019092', '2019351', '2019097', '2019098', '2019099', '2019358', '2019106',
                     '2019107', '2019108', '2019109', '2019365', '2019111', '2019113', '2019117',
                     '2019118', '2019120', '2019121', '2019122', '2019123', '2019127', '2020133',
                     '2020007', '2020140', '2020014', '2020147', '2020021', '2020154', '2020028',
                     '2020161', '2020035', '2020168', '2020042', '2020175', '2020049', '2020182',
                     '2020056', '2020063', '2020070', '2020077', '2020336', '2020084', '2020343',
                     '2020091', '2020350', '2020098', '2020357', '2020105', '2020364', '2020112',
                     '2020119', '2020126', '2021131', '2021005', '2021138', '2021012', '2021145',
                     '2021019', '2021152', '2021026', '2021159', '2021033', '2021166', '2021040',
                     '2021173', '2021047', '2021180', '2021054', '2021061', '2021068', '2021075',
                     '2021082', '2021089', '2021096', '2021103', '2021110', '2021117', '2021124',
                     ]

pathwheretosave = 'C:\proje'
modis_account_name = 'modissa'
modis_container_name = 'modis-006'
modis_account_url = 'https://' + modis_account_name + '.blob.core.windows.net/'
modis_blob_root = modis_account_url + modis_container_name + '/'
modis_tile_extents_url = modis_blob_root + 'sn_bound_10deg.txt'

modis_container_client = ContainerClient(account_url=modis_account_url,
                                         container_name=modis_container_name,
                                         credential=None)


def list_blobs_in_folder(container_name, folder_name):
    """
    List all blobs in a virtual folder in an Azure blob container
    """

    files = []
    generator = modis_container_client.list_blobs(name_starts_with=folder_name)
    for blob in generator:
        files.append(blob.name)
    return files


def list_hdf_blobs_in_folder(container_name, folder_name):
    """"
    List .hdf files in a folder
    """

    files = list_blobs_in_folder(container_name, folder_name)
    files = [fn for fn in files if fn.endswith('.hdf')]
    return files


def download(output_path, how='everything', ):
    product = 'MOD10A1'
    tt = [[8, 4], [8, 5], [9, 4], [9, 5], [10, 4]]

    if how == 'everything':
        for pair in tt:
            h = pair[0]
            v = pair[1]
            for daynum in temporalr_archive:
                folder = product + '/' + '{:0>2d}/{:0>2d}'.format(h, v) + '/' + daynum

                # Find all HDF files from this tile on this day
                filenames = list_hdf_blobs_in_folder(modis_container_name, folder)
                # print('Found {} matching file(s):'.format(len(filenames)))
                for fn in filenames:
                    print(fn)

                # Work with the first returned URL
                if len(filenames) == 0:
                    pass
                else:
                    blob_name = filenames[0]

                    # Download to a temporary file
                    url = modis_blob_root + blob_name
                    path = str(output_path + '/fol' + daynum[:4])
                    if not os.path.exists(path):
                        os.mkdir(path)
                    filename = os.path.join(path, blob_name.split('/')[-1])
                    print(filename)
                    if not os.path.isfile(filename):
                        wget.download(url, filename)

    if how == 'new':
        for pair in tt:
            h = pair[0]
            v = pair[1]
            for daynum in temporalr:
                folder = product + '/' + '{:0>2d}/{:0>2d}'.format(h, v) + '/' + daynum

                # Find all HDF files from this tile on this day
                filenames = list_hdf_blobs_in_folder(modis_container_name, folder)
                # print('Found {} matching file(s):'.format(len(filenames)))
                for fn in filenames:
                    print(fn)

                # Work with the first returned URL
                if len(filenames) == 0:
                    pass
                else:
                    blob_name = filenames[0]
                    url = modis_blob_root + blob_name
                    path = Path(str(output_path + '/fol' + daynum[:4]))

                    if not os.path.exists(path):
                        os.makedirs(path)
                    filename = os.path.join(path, blob_name.split('/')[-1])
                    print(filename)
                    if not os.path.isfile(filename):
                        wget.download(url, filename)
