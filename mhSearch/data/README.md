# Dataset Description

## Pointsmapping.ods

A three column spreadsheet (ID,X,Y) which points mapping in local coordinates.
Each ID represents an unique place on the map. The X-Y coordinates represents the local coordinates.

## For each measure

__measure1(2)_timestamp_id.csv:__

Timestamp (Unixtime) of arrival on placeID, timestamp (Unixtime) of departure by placeID, Place ID identifier (0-324)

__measure1(2)_smartphone_sens.csv:__

According to measure1(2)_timestamp_id.csv, this csv contains the data sensors retrieved by the smartphone.
Timestamp, AccelerationX, AccelerationY, AccelerationZ, MagneticFieldX, MagneticFieldY, MagneticFieldZ, Z-AxisAgle(Azimuth), X-AxisAngle(Pitch), Y-AxisAngle(Roll), GyroX, GyroY, GyroZ

__measure1(2)_smartwatch_sens.csv:__

According to measure1(2)_timestamp_id.csv, this csv contains the data sensors retrieved by the smartwatch. 
Timestamp, AccelerationX, AccelerationY, AccelerationZ, MagneticFieldX, MagneticFieldY, MagneticFieldZ, Z-AxisAgle(Azimuth), X-AxisAngle(Pitch), Y-AxisAngle(Roll), GyroX, GyroY, GyroZ

__measure1(2)_smartphone_wifi.csv:__

Each rows contains PlaceId (ascending order) and 127 column, with RSSI level for each different
WAPs retrieved during the campaign. Not all the WAPs are detected in each scan.
For these WAPs, the articial RSSI value is -100 (dbm).

## Citation Requests

Barsocchi, P., Crivello, A., La Rosa, D., & Palumbo, F. (2016, October). A multisource and multivariate dataset for indoor localization methods based on WLAN and geo-magnetic field fingerprinting. In Indoor Positioning and Indoor Navigation (IPIN), 2016 International Conference on (pp. 1-8). IEEE.

__Relevant Paper__
https://ieeexplore.ieee.org/document/7743678/
__Url Data__
https://archive.ics.uci.edu/ml/datasets/Geo-Magnetic+field+and+WLAN+dataset+for+indoor+localisation+from+wristband+and+smartphone#