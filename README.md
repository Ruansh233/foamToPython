This is an ongoing repository to read OpenFOAM data into numpy array.
It is faster than [FluidFoam](https://github.com/fluiddyn/fluidfoam), [PyFoam](https://pypi.org/project/PyFoam/), and [foamlib](https://github.com/gerlero/foamlib) in reading fields.
However, the aforementioned packages have more functions compared to this one.

### <span style="color:blue;">OFField</span> class
The class could read OpenFOAM fields. 
It can process volScalarField or volVectorField, either uniform or non-uniform internalField.

#### read field, e.g., U
`U = readOFData.OFField('case/1/U', 'vector', read_data=True)`

The arguments are:
1. _filename_: the path of the field file.
2. _data_type_: the type of the field, e.g., "scalar", "vector", "label".
3. _read_data_: whether to read the field when initializing the class.
4. _parallel_: whether the case is run in parallel.

If read_data is False, the field will not be read when initializing the class. You can use `U._readField()` to read it later, or access `U.internalField` or `U.boundaryField`, which will trigger the reading of the field.

The package can read parallel case, e.g., `U = readOFData.OFField('case/1/U', 'vector', parallel=True)`.

#### load <span style="color:blue;">internalField</span> and <span style="color:blue;">boundaryField</span>

`U_cell = U.internalField`
`U_boundary = U.boundaryField`

1. _U_cell_ is a numpy array, store the fields values. The length is 1 for the uniform internal field.
2. _U_boundary_ is a dict store each patches. For each patch, it contain _type_ for the type of boundary. If the _type_ is _fixedValue_, the numpy array can be accessed by the key _value_. For example: `U.boundaryField['velocityInlet']['value']`

For parallel case, both internalField and boundaryField are read from all processors and stored as lists. For example, `U.internalField[0]` is the internalField from processor 0.

#### <span style="color:blue;">writeField</span>
You can modify the data of _U_ and then write it as OF field. 

The arguments are:
1. _path_: the path to write the field. It can contain `<timeDir>` and `<fieldName>`, which will be replaced by the arguments `timeDir` and `fieldName` if provided.
2. _timeDir_: the time directory to write the field. Default is `None`.
3. _fieldName_: the name of the field to write. Default is `None`.

Same function can be used to write parallel case, e.g., `U.writeField('case/<timeDir>/<fieldName>')`.

Therefore, you can use two types of inputs, e.g.,
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

The arguments are:
1. _U_: the OFField class instance, which contains the data to perform POD.
2. _POD_algo_: the algorithm to perform POD, can be "svd" or "eigen". Default is "eigen".
3. _rank_: the number of modes to compute. Default is 10, which means 10 modes are computed.

The modes can be exported with OpenFOAM format using
`pod.writeModes(outputDir, fieldName=<fieldName>)`.

The arguments are:
1. _outputDir_: the directory to write the modes. The modes will be written in folders `outputDir/1`, `outputDir/2`, ..., `outputDir/rank`.
2. _fieldName_: the name of the field to write. Default is `None`.

### Parallel case
The package can read and write parallel case. 
However, the speed is slower than the serial case, and it will be improved in the future.