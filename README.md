This is an ongoing repository to read OpenFOAM data into numpy array.

### readList function
The function could read field value like velocity and pressure.
__It is not capable to read uniform fields.__
The data type are: "lable", "scalar", "vector".

For example: U = readList("1/U", "vector").

### readListList function
The function could read ListList, like the cellZones file.
The data type are: "lable", "scalar", "vector".

For example: cellZones = readList("constant/polyMesh/cellZones", "label").