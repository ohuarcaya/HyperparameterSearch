Copyright (c) 2018, Universitat Jaume I (UJI)
These data is licensed under CC Attribution 4.0 International (CC BY).
This documentation is licensed under CC0 license.

The folders contained in this folder contains the data related to the BLE RSS fingerprint collection and related information described in TODO and available at Zenodo repository, DOI 10.5281/zenodo.1618692. The collection was performed in two zones, and the name of a file includes an indication of which zone its information refers to: the Library area 'lib' or the Geotec area 'geo'. The files are organized into three subfolders: 'dep' for BLE beacon deployment details, 'obs' for obstacles (shelves and pillars) found inside the collection area, and 'rss' for BLE RSS fingerprint collection data. For BLE RSS fingerprint files:

* The collection for each zone is represented by four files: the RSS, the time, the coordinates, and the identifiers files, so that the ith row of each of them holds the respective information of the ith fingerprint collection in the zone. The file name points whether the files contains rss measurement values ('rss'), coordinates ('crd'), time information ('tms'), or fingerprint identification information ('ids').
* Each column in the RSS file and the time file represents the intensity measurement values (dBm) and their collection time for a specific BLE beacon. The two zones have no common beacons. If the beacon was not detected for a fingerprint, its intensity value was set to 100 and its collection time to zero for that fingerprint.
