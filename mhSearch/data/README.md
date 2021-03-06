# Dataset Description

## Data Set Information

Many real world applications need to know the localization of a user in the world to provide their services. Therefore, automatic user localization has been a hot research topic in the last years. Automatic user localization consists of estimating the position of the user (latitude, longitude and altitude) by using an electronic device, usually a mobile phone. Outdoor localization problem can be solved very accurately thanks to the inclusion of GPS sensors into the mobile devices. However, indoor localization is still an open problem mainly due to the loss of GPS signal in indoor environments. Although, there are some indoor positioning technologies and methodologies, this database is focused on WLAN fingerprint-based ones (also know as WiFi Fingerprinting).

Although there are many papers in the literature trying to solve the indoor localization problem using a WLAN fingerprint-based method, there still exists one important drawback in this field which is the lack of a common database for comparison purposes. So, UJIIndoorLoc database is presented to overcome this gap. We expect that the proposed database will become the reference database to compare different indoor localization methodologies based on WiFi fingerprinting.

The UJIIndoorLoc database covers three buildings of Universitat Jaume I with 4 or more floors and almost 110.000m2. It can be used for classification, e.g. actual building and floor identification, or regression, e.g. actual longitude and latitude estimation. It was created in 2013 by means of more than 20 different users and 25 Android devices. The database consists of 19937 training/reference records (trainingData.csv file) and 1111 validation/test records (validationData.csv file).

The 529 attributes contain the WiFi fingerprint, the coordinates where it was taken, and other useful information.

Each WiFi fingerprint can be characterized by the detected Wireless Access Points (WAPs) and the corresponding Received Signal Strength Intensity (RSSI). The intensity values are represented as negative integer values ranging -104dBm (extremely poor signal) to 0dbM. The positive value 100 is used to denote when a WAP was not detected. During the database creation, 520 different WAPs were detected. Thus, the WiFi fingerprint is composed by 520 intensity values.

Then the coordinates (latitude, longitude, floor) and Building ID are provided as the attributes to be predicted.

Additional information has been provided.

The particular space (offices, labs, etc.) and the relative position (inside/outside the space) where the capture was taken have been recorded. Outside means that the capture was taken in front of the door of the space.

Information about who (user), how (android device & version) and when (timestamp) WiFi capture was taken is also recorded.

## Attribute Information

- __Attribute 001 (WAP001):__ Intensity value for WAP001. Negative integer values from -104 to 0 and +100. Positive value 100 used if WAP001 was not detected.
....
- __Attribute 520 (WAP520):__ Intensity value for WAP520. Negative integer values from -104 to 0 and +100. Positive Vvalue 100 used if WAP520 was not detected.
- __Attribute 521 (Longitude):__ Longitude. Negative real values from -7695.9387549299299000 to -7299.786516730871000
- __Attribute 522 (Latitude):__ Latitude. Positive real values from 4864745.7450159714 to 4865017.3646842018.
- __Attribute 523 (Floor):__ Altitude in floors inside the building. Integer values from 0 to 4.
- __Attribute 524 (BuildingID):__ ID to identify the building. Measures were taken in three different buildings. Categorical integer values from 0 to 2.
- __Attribute 525 (SpaceID):__ Internal ID number to identify the Space (office, corridor, classroom) where the capture was taken. Categorical integer values.
- __Attribute 526 (RelativePosition):__ Relative position with respect to the Space (1 - Inside, 2 - Outside in Front of the door). Categorical integer values.
- __Attribute 527 (UserID):__ User identifier (see below). Categorical integer values.
- __Attribute 528 (PhoneID):__ Android device identifier (see below). Categorical integer values.
- __Attribute 529 (Timestamp):__ UNIX Time when the capture was taken. Integer value.

### UserID Anonymized user Height (cm)

- 0 USER0000 (Validation User) N/A
- 1 USER0001 170
- 2 USER0002 176
- 3 USER0003 172
- 4 USER0004 174
- 5 USER0005 184
- 6 USER0006 180
- 7 USER0007 160
- 8 USER0008 176
- 9 USER0009 177
- 10 USER0010 186
- 11 USER0011 176
- 12 USER0012 158
- 13 USER0013 174
- 14 USER0014 173
- 15 USER0015 174
- 16 USER0016 171
- 17 USER0017 166
- 18 USER0018 162

### PhoneID Android Device Android Ver. UserID

- 0 Celkon A27 4.0.4(6577) 0            [120]
- 1 GT-I8160 2.3.6 8                    [507]
- 2 GT-I8160 4.1.2 0                    [52]
- 3 GT-I9100 4.0.4 5                    [610]
- 4 GT-I9300 4.1.2 0                    [69]
- 5 GT-I9505 4.2.2 0                    [17]
- 6 GT-S5360 2.3.6 7                    [1383]
- 7 GT-S6500 2.3.6 14                   [1596]
- 8 Galaxy Nexus 4.2.2 10               [913]
- 9 Galaxy Nexus 4.3 0                  [77]
- 10 HTC Desire HD 2.3.5 18             [440]
- 11 HTC One 4.1.2 15                   [498]
- 12 HTC One 4.2.2 0                    [70]
- 13 HTC Wildfire S 2.3.5 0,11          [4885]
- 14 LT22i 4.0.4 0,1,9,16               [4863]
- 15 LT22i 4.1.2 0                      [36]
- 16 LT26i 4.0.4 3                      [192]
- 17 M1005D 4.0.4 13                    [841]
- 18 MT11i 2.3.4 4                      [374]
- 19 Nexus 4 4.2.2 6                    [980]
- 20 Nexus 4 4.3 0                      [213]
- 21 Nexus S 4.1.2 0                    [60]
- 22 Orange Monte Carlo 2.3.5 17        [724]
- 23 Transformer TF101 4.0.3 2          [1091]
- 24 bq Curie 4.1.1 12                  [437]

## Literatura

### Relevant Papers

Joaquín Torres-Sospedra, Raúl Montoliu, Adolfo Martínez-Usó, Tomar J. Arnau, Joan P. Avariento, Mauri Benedito-Bordonau, Joaquín Huerta
UJIIndoorLoc: A New Multi-building and Multi-floor Database for WLAN Fingerprint-based Indoor Localization Problems
In Proceedings of the Fifth International Conference on Indoor Positioning and Indoor Navigation, 2014.
Available at: [Web Link]

### Citation Request

Joaquín Torres-Sospedra, Raúl Montoliu, Adolfo Martínez-Usó, Tomar J. Arnau, Joan P. Avariento, Mauri Benedito-Bordonau, Joaquín Huerta
UJIIndoorLoc: A New Multi-building and Multi-floor Database for WLAN Fingerprint-based Indoor Localization Problems
In Proceedings of the Fifth International Conference on Indoor Positioning and Indoor Navigation, 2014.
Available at: [Web Link]

__Relevant Paper:__
<http://www.ipin2014.org/wp/pdf/4A-3.pdf>

__Url Data:__
<https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc#>