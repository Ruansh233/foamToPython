This is an ongoing repository to read OpenFOAM data into numpy array.
It is faster than __fluidfoam__, __PyFoam__, and __foamlib__ in reading fields. 
Note that, the aforementioned packages have more functions compared to this one.

### <span style="color:blue;">OFField</span> class
The class could read OpenFOAM fields. 
It can process volScalarField or volVectorField, either uniform or non-uniform internalField.

#### read field, e.g., U
`U = readOFData.OFField('case/1/U', 'vector')`

#### load <span style="color:blue;">internalField</span> and <span style="color:blue;">boundaryField</span>

`U_cell = U.internalField`
`U_boundary = U.boundaryField`

1. _U_cell_ is a numpy array, store the fields values. The length is 1 for the uniform internal field.
2. _U_boundary_ is a dict store each patches. For each patch, it contain _type_ for the type of boundary. If the _type_ is _fixedValue_, the numpy array can be accessed by the key _value_. For example: `U.boundaryField['velocityInlet']['value']`

#### <span style="color:blue;">writeField</span>
You can modify the data of _U_ and then write it as OF field. 
There are two types of inputs, e.g.,
1. `U.writeField('case/<timeDir>/<fieldName>')`.
2. `U.writeField('case/test', timeDir=<timeDir>, fieldName=<fieldName>)`. Note that `timeDir` and `fieldName` are needed when using **Paraview** to read the fields.

### <span style="color:blue;">readList</span> function
The function could read field value like velocity and pressure.
The data type are: "lable", "scalar", "vector".

For example: `U = readList("1/U", "vector").`

### <span style="color:blue;">readListList</span> function
The function could read ListList, like the cellZones file.
The data type are: "lable", "scalar", "vector".

For example: `cellZones = readList("constant/polyMesh/cellZones", "label")`.