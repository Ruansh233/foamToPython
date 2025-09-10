This is an ongoing repository to read OpenFOAM data into numpy array.
It is faster than [FluidFoam](https://github.com/fluiddyn/fluidfoam), [PyFoam](https://pypi.org/project/PyFoam/), and [foamlib](https://github.com/gerlero/foamlib) in reading fields.
However, the aforementioned packages have more functions compared to this one.

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

### Perform <span style="color:blue;">POD</span> to openfoam fields.
Please check the submodule _PODopenFOAM_ and the class under it _PODmodes_, which can be called `foamToPython.PODmodes`. It can be created as,

`pod = PODmodes(U, POD_algo=<POD_algo>, rank=<rank>)`.

The modes can be export with OpenFOAM style using
`pod.writeModes(outputDir, fieldName=<fieldName>)`.