ŞÓ
ďÂ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 

ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.11.12v2.11.0-94-ga3e2c692c188ë
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 * G
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 * žĆ
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *)\ˇA
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *\ż
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0

Adam/v/dense_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_69/bias
y
(Adam/v/dense_69/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_69/bias*
_output_shapes
:*
dtype0

Adam/m/dense_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_69/bias
y
(Adam/m/dense_69/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_69/bias*
_output_shapes
:*
dtype0

Adam/v/dense_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *'
shared_nameAdam/v/dense_69/kernel

*Adam/v/dense_69/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_69/kernel*
_output_shapes
:	 *
dtype0

Adam/m/dense_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *'
shared_nameAdam/m/dense_69/kernel

*Adam/m/dense_69/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_69/kernel*
_output_shapes
:	 *
dtype0

Adam/v/dense_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/dense_68/bias
z
(Adam/v/dense_68/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_68/bias*
_output_shapes	
: *
dtype0

Adam/m/dense_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/dense_68/bias
z
(Adam/m/dense_68/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_68/bias*
_output_shapes	
: *
dtype0

Adam/v/dense_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *'
shared_nameAdam/v/dense_68/kernel

*Adam/v/dense_68/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_68/kernel*
_output_shapes
:	 *
dtype0

Adam/m/dense_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *'
shared_nameAdam/m/dense_68/kernel

*Adam/m/dense_68/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_68/kernel*
_output_shapes
:	 *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
dense_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_69/bias
k
!dense_69/bias/Read/ReadVariableOpReadVariableOpdense_69/bias*
_output_shapes
:*
dtype0
{
dense_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 * 
shared_namedense_69/kernel
t
#dense_69/kernel/Read/ReadVariableOpReadVariableOpdense_69/kernel*
_output_shapes
:	 *
dtype0
s
dense_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_68/bias
l
!dense_68/bias/Read/ReadVariableOpReadVariableOpdense_68/bias*
_output_shapes	
: *
dtype0
{
dense_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 * 
shared_namedense_68/kernel
t
#dense_68/kernel/Read/ReadVariableOpReadVariableOpdense_68/kernel*
_output_shapes
:	 *
dtype0
s
serving_default_examplesPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
Ą
StatefulPartitionedCallStatefulPartitionedCallserving_default_examplesConst_3Const_2Const_1Constdense_68/kerneldense_68/biasdense_69/kerneldense_69/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_304629

NoOpNoOp
á2
Const_4Const"/device:CPU:0*
_output_shapes
: *
dtype0*2
value2B2 B2
ĺ
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer_with_weights-0
layer-13
layer_with_weights-1
layer-14
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	tft_layer

signatures*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
Ś
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
Ś
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
´
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
$6 _saved_model_loader_tracked_dict* 
 
&0
'1
.2
/3*
 
&0
'1
.2
/3*
* 
°
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
<trace_0
=trace_1
>trace_2
?trace_3* 
6
@trace_0
Atrace_1
Btrace_2
Ctrace_3* 
* 

D
_variables
E_iterations
F_learning_rate
G_index_dict
H
_momentums
I_velocities
J_update_step_xla*

Kserving_default* 
* 
* 
* 

Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Qtrace_0* 

Rtrace_0* 

&0
'1*

&0
'1*
* 

Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

Xtrace_0* 

Ytrace_0* 
_Y
VARIABLE_VALUEdense_68/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_68/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 

Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

_trace_0* 

`trace_0* 
_Y
VARIABLE_VALUEdense_69/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_69/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

ftrace_0
gtrace_1* 

htrace_0
itrace_1* 
t
j	_imported
k_wrapped_function
l_structured_inputs
m_structured_outputs
n_output_to_inputs_map* 
* 
z
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15*

o0
p1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
C
E0
q1
r2
s3
t4
u5
v6
w7
x8*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
q0
s1
u2
w3*
 
r0
t1
v2
x3*
* 
>
y	capture_0
z	capture_1
{	capture_2
|	capture_3* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
>
y	capture_0
z	capture_1
{	capture_2
|	capture_3* 
>
y	capture_0
z	capture_1
{	capture_2
|	capture_3* 
>
y	capture_0
z	capture_1
{	capture_2
|	capture_3* 
>
y	capture_0
z	capture_1
{	capture_2
|	capture_3* 
¨
}created_variables
~	resources
trackable_objects
initializers
assets

signatures
$_self_saveable_object_factories
ktransform_fn* 
>
y	capture_0
z	capture_1
{	capture_2
|	capture_3* 
* 
* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/dense_68/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_68/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_68/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_68/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_69/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_69/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_69/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_69/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 

serving_default* 
* 

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
>
y	capture_0
z	capture_1
{	capture_2
|	capture_3* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
É
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_68/kernel/Read/ReadVariableOp!dense_68/bias/Read/ReadVariableOp#dense_69/kernel/Read/ReadVariableOp!dense_69/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp*Adam/m/dense_68/kernel/Read/ReadVariableOp*Adam/v/dense_68/kernel/Read/ReadVariableOp(Adam/m/dense_68/bias/Read/ReadVariableOp(Adam/v/dense_68/bias/Read/ReadVariableOp*Adam/m/dense_69/kernel/Read/ReadVariableOp*Adam/v/dense_69/kernel/Read/ReadVariableOp(Adam/m/dense_69/bias/Read/ReadVariableOp(Adam/v/dense_69/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_305540
Ú
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_68/kerneldense_68/biasdense_69/kerneldense_69/bias	iterationlearning_rateAdam/m/dense_68/kernelAdam/v/dense_68/kernelAdam/m/dense_68/biasAdam/v/dense_68/biasAdam/m/dense_69/kernelAdam/v/dense_69/kernelAdam/m/dense_69/biasAdam/v/dense_69/biastotal_1count_1totalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_305604Ů
á	

$__inference_signature_wrapper_304629
examples
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3:	 
	unknown_4:	 
	unknown_5:	 
	unknown_6:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_serve_tf_examples_fn_304606o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ď(
­
9__inference_transform_features_layer_layer_call_fn_305396%
!inputs_avg_cars_at_home_approx__1
inputs_coffee_bar
inputs_cost
inputs_florist
inputs_low_fat
inputs_num_children_at_home
inputs_prepared_food
inputs_salad_bar#
inputs_store_sales_in_millions_
inputs_store_sqft
inputs_total_children"
inputs_unit_sales_in_millions_
inputs_video_store
unknown
	unknown_0
	unknown_1
	unknown_2
identity	

identity_1	

identity_2	

identity_3	

identity_4	

identity_5	

identity_6	

identity_7

identity_8

identity_9	
identity_10	
identity_11	˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall!inputs_avg_cars_at_home_approx__1inputs_coffee_barinputs_costinputs_floristinputs_low_fatinputs_num_children_at_homeinputs_prepared_foodinputs_salad_barinputs_store_sales_in_millions_inputs_store_sqftinputs_total_childreninputs_unit_sales_in_millions_inputs_video_storeunknown	unknown_0	unknown_1	unknown_2*
Tin
2*
Tout
2										*
_collective_manager_ids
 *ú
_output_shapesç
ä:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_304740o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_9Identity StatefulPartitionedCall:output:9^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
Identity_10Identity!StatefulPartitionedCall:output:10^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
Identity_11Identity!StatefulPartitionedCall:output:11^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
;
_user_specified_name#!inputs_avg_cars_at_home_approx__1:ZV
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_coffee_bar:TP
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_cost:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameinputs_florist:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameinputs_low_fat:d`
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs_num_children_at_home:]Y
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_prepared_food:YU
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_salad_bar:hd
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
9
_user_specified_name!inputs_store_sales_in_millions_:Z	V
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_store_sqft:^
Z
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_total_children:gc
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name inputs_unit_sales_in_millions_:[W
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_video_store:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
š

I__inference_concatenate_4_layer_call_and_return_conditional_losses_305309
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ý
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11concat/axis:output:0*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ů
_input_shapesç
ä:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_8:Q	M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_9:R
N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_11
&
Ž
9__inference_transform_features_layer_layer_call_fn_304773
placeholder

coffee_bar
cost
florist
low_fat
num_children_at_home
prepared_food
	salad_bar
placeholder_1

store_sqft
total_children
placeholder_2
video_store
unknown
	unknown_0
	unknown_1
	unknown_2
identity	

identity_1	

identity_2	

identity_3	

identity_4	

identity_5	

identity_6	

identity_7

identity_8

identity_9	
identity_10	
identity_11	˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallplaceholder
coffee_barcostfloristlow_fatnum_children_at_homeprepared_food	salad_barplaceholder_1
store_sqfttotal_childrenplaceholder_2video_storeunknown	unknown_0	unknown_1	unknown_2*
Tin
2*
Tout
2										*
_collective_manager_ids
 *ú
_output_shapesç
ä:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_304740o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_9Identity StatefulPartitionedCall:output:9^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
Identity_10Identity!StatefulPartitionedCall:output:10^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
Identity_11Identity!StatefulPartitionedCall:output:11^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameavg_cars_at home(approx).1:SO
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
coffee_bar:MI
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namecost:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	florist:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	low_fat:]Y
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namenum_children_at_home:VR
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameprepared_food:RN
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	salad_bar:a]
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namestore_sales(in millions):S	O
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
store_sqft:W
S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nametotal_children:`\
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameunit_sales(in millions):TP
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namevideo_store:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ť,
Î
__inference__traced_save_305540
file_prefix.
*savev2_dense_68_kernel_read_readvariableop,
(savev2_dense_68_bias_read_readvariableop.
*savev2_dense_69_kernel_read_readvariableop,
(savev2_dense_69_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop5
1savev2_adam_m_dense_68_kernel_read_readvariableop5
1savev2_adam_v_dense_68_kernel_read_readvariableop3
/savev2_adam_m_dense_68_bias_read_readvariableop3
/savev2_adam_v_dense_68_bias_read_readvariableop5
1savev2_adam_m_dense_69_kernel_read_readvariableop5
1savev2_adam_v_dense_69_kernel_read_readvariableop3
/savev2_adam_m_dense_69_bias_read_readvariableop3
/savev2_adam_v_dense_69_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const_4

identity_1˘MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ˝
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ć
valueÜBŮB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_68_kernel_read_readvariableop(savev2_dense_68_bias_read_readvariableop*savev2_dense_69_kernel_read_readvariableop(savev2_dense_69_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop1savev2_adam_m_dense_68_kernel_read_readvariableop1savev2_adam_v_dense_68_kernel_read_readvariableop/savev2_adam_m_dense_68_bias_read_readvariableop/savev2_adam_v_dense_68_bias_read_readvariableop1savev2_adam_m_dense_69_kernel_read_readvariableop1savev2_adam_v_dense_69_kernel_read_readvariableop/savev2_adam_m_dense_69_bias_read_readvariableop/savev2_adam_v_dense_69_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const_4"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *!
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:ł
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes{
y: :	 : :	 :: : :	 :	 : : :	 :	 ::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	 :!

_output_shapes	
: :%!

_output_shapes
:	 : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 :%!

_output_shapes
:	 :!	

_output_shapes	
: :!


_output_shapes	
: :%!

_output_shapes
:	 :%!

_output_shapes
:	 : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Ż
(__inference_model_4_layer_call_fn_305112
placeholder
coffee_bar_xf

florist_xf

low_fat_xf
num_children_at_home_xf
prepared_food_xf
salad_bar_xf
placeholder_1
store_sqft_xf
total_children_xf
placeholder_2
video_store_xf
unknown:	 
	unknown_0:	 
	unknown_1:	 
	unknown_2:
identity˘StatefulPartitionedCall˛
StatefulPartitionedCallStatefulPartitionedCallplaceholdercoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfplaceholder_1store_sqft_xftotal_children_xfplaceholder_2video_store_xfunknown	unknown_0	unknown_1	unknown_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_305077o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesď
ě:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_nameavg_cars_at home(approx).1_xf:VR
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namecoffee_bar_xf:SO
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
florist_xf:SO
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
low_fat_xf:`\
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_namenum_children_at_home_xf:YU
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameprepared_food_xf:UQ
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namesalad_bar_xf:d`
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namestore_sales(in millions)_xf:VR
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namestore_sqft_xf:Z	V
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nametotal_children_xf:c
_
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameunit_sales(in millions)_xf:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namevideo_store_xf
Ł

÷
D__inference_dense_68_layer_call_and_return_conditional_losses_305329

inputs1
matmul_readvariableop_resource:	 .
biasadd_readvariableop_resource:	 
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ú

C__inference_model_4_layer_call_and_return_conditional_losses_305138
placeholder
coffee_bar_xf

florist_xf

low_fat_xf
num_children_at_home_xf
prepared_food_xf
salad_bar_xf
placeholder_1
store_sqft_xf
total_children_xf
placeholder_2
video_store_xf"
dense_68_305127:	 
dense_68_305129:	 "
dense_69_305132:	 
dense_69_305134:
identity˘ dense_68/StatefulPartitionedCall˘ dense_69/StatefulPartitionedCall
concatenate_4/PartitionedCallPartitionedCallplaceholdercoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfplaceholder_1store_sqft_xftotal_children_xfplaceholder_2video_store_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_304929
 dense_68/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_68_305127dense_68_305129*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_304942
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_305132dense_69_305134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_304959x
IdentityIdentity)dense_69/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesď
ě:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall:f b
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_nameavg_cars_at home(approx).1_xf:VR
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namecoffee_bar_xf:SO
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
florist_xf:SO
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
low_fat_xf:`\
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_namenum_children_at_home_xf:YU
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameprepared_food_xf:UQ
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namesalad_bar_xf:d`
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namestore_sales(in millions)_xf:VR
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namestore_sqft_xf:Z	V
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nametotal_children_xf:c
_
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameunit_sales(in millions)_xf:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namevideo_store_xf
Ô%
ł
C__inference_model_4_layer_call_and_return_conditional_losses_305245(
$inputs_avg_cars_at_home_approx__1_xf
inputs_coffee_bar_xf
inputs_florist_xf
inputs_low_fat_xf"
inputs_num_children_at_home_xf
inputs_prepared_food_xf
inputs_salad_bar_xf&
"inputs_store_sales_in_millions__xf
inputs_store_sqft_xf
inputs_total_children_xf%
!inputs_unit_sales_in_millions__xf
inputs_video_store_xf:
'dense_68_matmul_readvariableop_resource:	 7
(dense_68_biasadd_readvariableop_resource:	 :
'dense_69_matmul_readvariableop_resource:	 6
(dense_69_biasadd_readvariableop_resource:
identity˘dense_68/BiasAdd/ReadVariableOp˘dense_68/MatMul/ReadVariableOp˘dense_69/BiasAdd/ReadVariableOp˘dense_69/MatMul/ReadVariableOp[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :˝
concatenate_4/concatConcatV2$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xf"concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
dense_68/MatMulMatMulconcatenate_4/concat:output:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
dense_68/BiasAdd/ReadVariableOpReadVariableOp(dense_68_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0
dense_68/BiasAddBiasAdddense_68/MatMul:product:0'dense_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ c
dense_68/ReluReludense_68/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
dense_69/MatMulMatMuldense_68/Relu:activations:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
dense_69/SoftmaxSoftmaxdense_69/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentitydense_69/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ě
NoOpNoOp ^dense_68/BiasAdd/ReadVariableOp^dense_68/MatMul/ReadVariableOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesď
ě:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 2B
dense_68/BiasAdd/ReadVariableOpdense_68/BiasAdd/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp:m i
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
>
_user_specified_name&$inputs_avg_cars_at_home_approx__1_xf:]Y
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_coffee_bar_xf:ZV
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_florist_xf:ZV
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_low_fat_xf:gc
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name inputs_num_children_at_home_xf:`\
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_prepared_food_xf:\X
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_salad_bar_xf:kg
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
<
_user_specified_name$"inputs_store_sales_in_millions__xf:]Y
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_store_sqft_xf:a	]
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_nameinputs_total_children_xf:j
f
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
;
_user_specified_name#!inputs_unit_sales_in_millions__xf:^Z
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_video_store_xf
ç
°
(__inference_model_4_layer_call_fn_305190(
$inputs_avg_cars_at_home_approx__1_xf
inputs_coffee_bar_xf
inputs_florist_xf
inputs_low_fat_xf"
inputs_num_children_at_home_xf
inputs_prepared_food_xf
inputs_salad_bar_xf&
"inputs_store_sales_in_millions__xf
inputs_store_sqft_xf
inputs_total_children_xf%
!inputs_unit_sales_in_millions__xf
inputs_video_store_xf
unknown:	 
	unknown_0:	 
	unknown_1:	 
	unknown_2:
identity˘StatefulPartitionedCallł
StatefulPartitionedCallStatefulPartitionedCall$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xfunknown	unknown_0	unknown_1	unknown_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_304966o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesď
ě:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:m i
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
>
_user_specified_name&$inputs_avg_cars_at_home_approx__1_xf:]Y
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_coffee_bar_xf:ZV
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_florist_xf:ZV
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_low_fat_xf:gc
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name inputs_num_children_at_home_xf:`\
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_prepared_food_xf:\X
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_salad_bar_xf:kg
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
<
_user_specified_name$"inputs_store_sales_in_millions__xf:]Y
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_store_sqft_xf:a	]
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_nameinputs_total_children_xf:j
f
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
;
_user_specified_name#!inputs_unit_sales_in_millions__xf:^Z
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_video_store_xf
ĂN
Ü

"__inference__traced_restore_305604
file_prefix3
 assignvariableop_dense_68_kernel:	 /
 assignvariableop_1_dense_68_bias:	 5
"assignvariableop_2_dense_69_kernel:	 .
 assignvariableop_3_dense_69_bias:&
assignvariableop_4_iteration:	 *
 assignvariableop_5_learning_rate: <
)assignvariableop_6_adam_m_dense_68_kernel:	 <
)assignvariableop_7_adam_v_dense_68_kernel:	 6
'assignvariableop_8_adam_m_dense_68_bias:	 6
'assignvariableop_9_adam_v_dense_68_bias:	 =
*assignvariableop_10_adam_m_dense_69_kernel:	 =
*assignvariableop_11_adam_v_dense_69_kernel:	 6
(assignvariableop_12_adam_m_dense_69_bias:6
(assignvariableop_13_adam_v_dense_69_bias:%
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: 
identity_19˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9Ŕ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ć
valueÜBŮB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ý
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ł
AssignVariableOpAssignVariableOp assignvariableop_dense_68_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_68_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:š
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_69_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_69_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:ł
AssignVariableOp_4AssignVariableOpassignvariableop_4_iterationIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_5AssignVariableOp assignvariableop_5_learning_rateIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ŕ
AssignVariableOp_6AssignVariableOp)assignvariableop_6_adam_m_dense_68_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ŕ
AssignVariableOp_7AssignVariableOp)assignvariableop_7_adam_v_dense_68_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:ž
AssignVariableOp_8AssignVariableOp'assignvariableop_8_adam_m_dense_68_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ž
AssignVariableOp_9AssignVariableOp'assignvariableop_9_adam_v_dense_68_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ă
AssignVariableOp_10AssignVariableOp*assignvariableop_10_adam_m_dense_69_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ă
AssignVariableOp_11AssignVariableOp*assignvariableop_11_adam_v_dense_69_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_12AssignVariableOp(assignvariableop_12_adam_m_dense_69_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_v_dense_69_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:˛
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:˛
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Ű
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: Č
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ú

C__inference_model_4_layer_call_and_return_conditional_losses_305164
placeholder
coffee_bar_xf

florist_xf

low_fat_xf
num_children_at_home_xf
prepared_food_xf
salad_bar_xf
placeholder_1
store_sqft_xf
total_children_xf
placeholder_2
video_store_xf"
dense_68_305153:	 
dense_68_305155:	 "
dense_69_305158:	 
dense_69_305160:
identity˘ dense_68/StatefulPartitionedCall˘ dense_69/StatefulPartitionedCall
concatenate_4/PartitionedCallPartitionedCallplaceholdercoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfplaceholder_1store_sqft_xftotal_children_xfplaceholder_2video_store_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_304929
 dense_68/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_68_305153dense_68_305155*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_304942
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_305158dense_69_305160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_304959x
IdentityIdentity)dense_69/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesď
ě:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall:f b
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_nameavg_cars_at home(approx).1_xf:VR
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namecoffee_bar_xf:SO
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
florist_xf:SO
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
low_fat_xf:`\
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_namenum_children_at_home_xf:YU
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameprepared_food_xf:UQ
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namesalad_bar_xf:d`
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namestore_sales(in millions)_xf:VR
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namestore_sqft_xf:Z	V
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nametotal_children_xf:c
_
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameunit_sales(in millions)_xf:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namevideo_store_xf
¤

ö
D__inference_dense_69_layer_call_and_return_conditional_losses_304959

inputs1
matmul_readvariableop_resource:	 -
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
°6
Č
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_305459%
!inputs_avg_cars_at_home_approx__1
inputs_coffee_bar
inputs_cost
inputs_florist
inputs_low_fat
inputs_num_children_at_home
inputs_prepared_food
inputs_salad_bar#
inputs_store_sales_in_millions_
inputs_store_sqft
inputs_total_children"
inputs_unit_sales_in_millions_
inputs_video_store
unknown
	unknown_0
	unknown_1
	unknown_2
identity	

identity_1	

identity_2	

identity_3	

identity_4	

identity_5	

identity_6	

identity_7

identity_8

identity_9	
identity_10	
identity_11	˘StatefulPartitionedCallV
ShapeShape!inputs_avg_cars_at_home_approx__1*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Shape_1Shape!inputs_avg_cars_at_home_approx__1*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙ă
StatefulPartitionedCallStatefulPartitionedCall!inputs_avg_cars_at_home_approx__1inputs_coffee_barinputs_costPlaceholderWithDefault:output:0inputs_floristinputs_low_fatinputs_num_children_at_homeinputs_prepared_foodinputs_salad_barinputs_store_sales_in_millions_inputs_store_sqftinputs_total_childreninputs_unit_sales_in_millions_inputs_video_storeunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2											*
_output_shapesú
÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *"
fR
__inference_pruned_304423o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_2Identity StatefulPartitionedCall:output:3^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_3Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_6Identity StatefulPartitionedCall:output:7^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_7Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_8Identity StatefulPartitionedCall:output:9^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_9Identity!StatefulPartitionedCall:output:10^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
Identity_10Identity!StatefulPartitionedCall:output:11^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
Identity_11Identity!StatefulPartitionedCall:output:12^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
;
_user_specified_name#!inputs_avg_cars_at_home_approx__1:ZV
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_coffee_bar:TP
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_cost:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameinputs_florist:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameinputs_low_fat:d`
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs_num_children_at_home:]Y
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_prepared_food:YU
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_salad_bar:hd
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
9
_user_specified_name!inputs_store_sales_in_millions_:Z	V
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_store_sqft:^
Z
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_total_children:gc
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name inputs_unit_sales_in_millions_:[W
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_video_store:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ĺ

)__inference_dense_69_layer_call_fn_305338

inputs
unknown:	 
	unknown_0:
identity˘StatefulPartitionedCallŮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_304959o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙ : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ó1
¤
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_304740

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
unknown
	unknown_0
	unknown_1
	unknown_2
identity	

identity_1	

identity_2	

identity_3	

identity_4	

identity_5	

identity_6	

identity_7

identity_8

identity_9	
identity_10	
identity_11	˘StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙ż
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2PlaceholderWithDefault:output:0inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12unknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2											*
_output_shapesú
÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *"
fR
__inference_pruned_304423o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_2Identity StatefulPartitionedCall:output:3^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_3Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_6Identity StatefulPartitionedCall:output:7^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_7Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_8Identity StatefulPartitionedCall:output:9^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_9Identity!StatefulPartitionedCall:output:10^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
Identity_10Identity!StatefulPartitionedCall:output:11^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
Identity_11Identity!StatefulPartitionedCall:output:12^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:O	K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:O
K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ć
Ë
C__inference_model_4_layer_call_and_return_conditional_losses_304966

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11"
dense_68_304943:	 
dense_68_304945:	 "
dense_69_304960:	 
dense_69_304962:
identity˘ dense_68/StatefulPartitionedCall˘ dense_69/StatefulPartitionedCall˝
concatenate_4/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_304929
 dense_68/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_68_304943dense_68_304945*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_304942
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_304960dense_69_304962*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_304959x
IdentityIdentity)dense_69/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesď
ě:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:O	K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:O
K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Ż
(__inference_model_4_layer_call_fn_304977
placeholder
coffee_bar_xf

florist_xf

low_fat_xf
num_children_at_home_xf
prepared_food_xf
salad_bar_xf
placeholder_1
store_sqft_xf
total_children_xf
placeholder_2
video_store_xf
unknown:	 
	unknown_0:	 
	unknown_1:	 
	unknown_2:
identity˘StatefulPartitionedCall˛
StatefulPartitionedCallStatefulPartitionedCallplaceholdercoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfplaceholder_1store_sqft_xftotal_children_xfplaceholder_2video_store_xfunknown	unknown_0	unknown_1	unknown_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_304966o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesď
ě:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_nameavg_cars_at home(approx).1_xf:VR
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namecoffee_bar_xf:SO
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
florist_xf:SO
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
low_fat_xf:`\
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_namenum_children_at_home_xf:YU
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameprepared_food_xf:UQ
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namesalad_bar_xf:d`
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namestore_sales(in millions)_xf:VR
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namestore_sqft_xf:Z	V
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nametotal_children_xf:c
_
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameunit_sales(in millions)_xf:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namevideo_store_xf
Ć

)__inference_dense_68_layer_call_fn_305318

inputs
unknown:	 
	unknown_0:	 
identity˘StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_304942p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ť3
É
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_304883
placeholder

coffee_bar
cost
florist
low_fat
num_children_at_home
prepared_food
	salad_bar
placeholder_1

store_sqft
total_children
placeholder_2
video_store
unknown
	unknown_0
	unknown_1
	unknown_2
identity	

identity_1	

identity_2	

identity_3	

identity_4	

identity_5	

identity_6	

identity_7

identity_8

identity_9	
identity_10	
identity_11	˘StatefulPartitionedCall@
ShapeShapeplaceholder*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskB
Shape_1Shapeplaceholder*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙ä
StatefulPartitionedCallStatefulPartitionedCallplaceholder
coffee_barcostPlaceholderWithDefault:output:0floristlow_fatnum_children_at_homeprepared_food	salad_barplaceholder_1
store_sqfttotal_childrenplaceholder_2video_storeunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2											*
_output_shapesú
÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *"
fR
__inference_pruned_304423o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_2Identity StatefulPartitionedCall:output:3^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_3Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_6Identity StatefulPartitionedCall:output:7^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_7Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_8Identity StatefulPartitionedCall:output:9^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_9Identity!StatefulPartitionedCall:output:10^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
Identity_10Identity!StatefulPartitionedCall:output:11^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
Identity_11Identity!StatefulPartitionedCall:output:12^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameavg_cars_at home(approx).1:SO
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
coffee_bar:MI
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namecost:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	florist:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	low_fat:]Y
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namenum_children_at_home:VR
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameprepared_food:RN
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	salad_bar:a]
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namestore_sales(in millions):S	O
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
store_sqft:W
S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nametotal_children:`\
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameunit_sales(in millions):TP
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namevideo_store:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


I__inference_concatenate_4_layer_call_and_return_conditional_losses_304929

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ű
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11concat/axis:output:0*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ů
_input_shapesç
ä:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:O	K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:O
K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ç
°
(__inference_model_4_layer_call_fn_305214(
$inputs_avg_cars_at_home_approx__1_xf
inputs_coffee_bar_xf
inputs_florist_xf
inputs_low_fat_xf"
inputs_num_children_at_home_xf
inputs_prepared_food_xf
inputs_salad_bar_xf&
"inputs_store_sales_in_millions__xf
inputs_store_sqft_xf
inputs_total_children_xf%
!inputs_unit_sales_in_millions__xf
inputs_video_store_xf
unknown:	 
	unknown_0:	 
	unknown_1:	 
	unknown_2:
identity˘StatefulPartitionedCallł
StatefulPartitionedCallStatefulPartitionedCall$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xfunknown	unknown_0	unknown_1	unknown_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_305077o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesď
ě:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:m i
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
>
_user_specified_name&$inputs_avg_cars_at_home_approx__1_xf:]Y
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_coffee_bar_xf:ZV
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_florist_xf:ZV
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_low_fat_xf:gc
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name inputs_num_children_at_home_xf:`\
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_prepared_food_xf:\X
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_salad_bar_xf:kg
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
<
_user_specified_name$"inputs_store_sales_in_millions__xf:]Y
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_store_sqft_xf:a	]
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_nameinputs_total_children_xf:j
f
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
;
_user_specified_name#!inputs_unit_sales_in_millions__xf:^Z
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_video_store_xf
Ü%
Đ
!__inference__wrapped_model_304661
placeholder
coffee_bar_xf

florist_xf

low_fat_xf
num_children_at_home_xf
prepared_food_xf
salad_bar_xf
placeholder_1
store_sqft_xf
total_children_xf
placeholder_2
video_store_xfB
/model_4_dense_68_matmul_readvariableop_resource:	 ?
0model_4_dense_68_biasadd_readvariableop_resource:	 B
/model_4_dense_69_matmul_readvariableop_resource:	 >
0model_4_dense_69_biasadd_readvariableop_resource:
identity˘'model_4/dense_68/BiasAdd/ReadVariableOp˘&model_4/dense_68/MatMul/ReadVariableOp˘'model_4/dense_69/BiasAdd/ReadVariableOp˘&model_4/dense_69/MatMul/ReadVariableOpc
!model_4/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ě
model_4/concatenate_4/concatConcatV2placeholdercoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfplaceholder_1store_sqft_xftotal_children_xfplaceholder_2video_store_xf*model_4/concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
&model_4/dense_68/MatMul/ReadVariableOpReadVariableOp/model_4_dense_68_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0Ť
model_4/dense_68/MatMulMatMul%model_4/concatenate_4/concat:output:0.model_4/dense_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
'model_4/dense_68/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_68_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0Ş
model_4/dense_68/BiasAddBiasAdd!model_4/dense_68/MatMul:product:0/model_4/dense_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ s
model_4/dense_68/ReluRelu!model_4/dense_68/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
&model_4/dense_69/MatMul/ReadVariableOpReadVariableOp/model_4_dense_69_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0¨
model_4/dense_69/MatMulMatMul#model_4/dense_68/Relu:activations:0.model_4/dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'model_4/dense_69/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_69_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Š
model_4/dense_69/BiasAddBiasAdd!model_4/dense_69/MatMul:product:0/model_4/dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙x
model_4/dense_69/SoftmaxSoftmax!model_4/dense_69/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q
IdentityIdentity"model_4/dense_69/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ě
NoOpNoOp(^model_4/dense_68/BiasAdd/ReadVariableOp'^model_4/dense_68/MatMul/ReadVariableOp(^model_4/dense_69/BiasAdd/ReadVariableOp'^model_4/dense_69/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesď
ě:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 2R
'model_4/dense_68/BiasAdd/ReadVariableOp'model_4/dense_68/BiasAdd/ReadVariableOp2P
&model_4/dense_68/MatMul/ReadVariableOp&model_4/dense_68/MatMul/ReadVariableOp2R
'model_4/dense_69/BiasAdd/ReadVariableOp'model_4/dense_69/BiasAdd/ReadVariableOp2P
&model_4/dense_69/MatMul/ReadVariableOp&model_4/dense_69/MatMul/ReadVariableOp:f b
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_nameavg_cars_at home(approx).1_xf:VR
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namecoffee_bar_xf:SO
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
florist_xf:SO
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
low_fat_xf:`\
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_namenum_children_at_home_xf:YU
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameprepared_food_xf:UQ
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namesalad_bar_xf:d`
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namestore_sales(in millions)_xf:VR
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namestore_sqft_xf:Z	V
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nametotal_children_xf:c
_
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameunit_sales(in millions)_xf:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namevideo_store_xf
Ô%
ł
C__inference_model_4_layer_call_and_return_conditional_losses_305276(
$inputs_avg_cars_at_home_approx__1_xf
inputs_coffee_bar_xf
inputs_florist_xf
inputs_low_fat_xf"
inputs_num_children_at_home_xf
inputs_prepared_food_xf
inputs_salad_bar_xf&
"inputs_store_sales_in_millions__xf
inputs_store_sqft_xf
inputs_total_children_xf%
!inputs_unit_sales_in_millions__xf
inputs_video_store_xf:
'dense_68_matmul_readvariableop_resource:	 7
(dense_68_biasadd_readvariableop_resource:	 :
'dense_69_matmul_readvariableop_resource:	 6
(dense_69_biasadd_readvariableop_resource:
identity˘dense_68/BiasAdd/ReadVariableOp˘dense_68/MatMul/ReadVariableOp˘dense_69/BiasAdd/ReadVariableOp˘dense_69/MatMul/ReadVariableOp[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :˝
concatenate_4/concatConcatV2$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xf"concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
dense_68/MatMulMatMulconcatenate_4/concat:output:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
dense_68/BiasAdd/ReadVariableOpReadVariableOp(dense_68_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0
dense_68/BiasAddBiasAdddense_68/MatMul:product:0'dense_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ c
dense_68/ReluReludense_68/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
dense_69/MatMulMatMuldense_68/Relu:activations:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
dense_69/SoftmaxSoftmaxdense_69/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentitydense_69/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ě
NoOpNoOp ^dense_68/BiasAdd/ReadVariableOp^dense_68/MatMul/ReadVariableOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesď
ě:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 2B
dense_68/BiasAdd/ReadVariableOpdense_68/BiasAdd/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp:m i
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
>
_user_specified_name&$inputs_avg_cars_at_home_approx__1_xf:]Y
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_coffee_bar_xf:ZV
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_florist_xf:ZV
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_low_fat_xf:gc
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name inputs_num_children_at_home_xf:`\
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_prepared_food_xf:\X
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_salad_bar_xf:kg
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
<
_user_specified_name$"inputs_store_sales_in_millions__xf:]Y
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_store_sqft_xf:a	]
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_nameinputs_total_children_xf:j
f
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
;
_user_specified_name#!inputs_unit_sales_in_millions__xf:^Z
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_video_store_xf
ô^
ž
'__inference_serve_tf_examples_fn_304606
examples#
transform_features_layer_304558#
transform_features_layer_304560#
transform_features_layer_304562#
transform_features_layer_304564B
/model_4_dense_68_matmul_readvariableop_resource:	 ?
0model_4_dense_68_biasadd_readvariableop_resource:	 B
/model_4_dense_69_matmul_readvariableop_resource:	 >
0model_4_dense_69_biasadd_readvariableop_resource:
identity˘'model_4/dense_68/BiasAdd/ReadVariableOp˘&model_4/dense_68/MatMul/ReadVariableOp˘'model_4/dense_69/BiasAdd/ReadVariableOp˘&model_4/dense_69/MatMul/ReadVariableOp˘0transform_features_layer/StatefulPartitionedCallU
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_1Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_2Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_3Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_4Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_5Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_6Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_7Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_8Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_9Const*
_output_shapes
: *
dtype0*
valueB X
ParseExample/Const_10Const*
_output_shapes
: *
dtype0*
valueB X
ParseExample/Const_11Const*
_output_shapes
: *
dtype0*
valueB X
ParseExample/Const_12Const*
_output_shapes
: *
dtype0*
valueB d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB ź
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*á
value×BÔBavg_cars_at home(approx).1B
coffee_barBcostBfloristBlow_fatBnum_children_at_homeBprepared_foodB	salad_barBstore_sales(in millions)B
store_sqftBtotal_childrenBunit_sales(in millions)Bvideo_storej
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB ő
ParseExample/ParseExampleV2ParseExampleV2examples*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0ParseExample/Const_1:output:0ParseExample/Const_2:output:0ParseExample/Const_3:output:0ParseExample/Const_4:output:0ParseExample/Const_5:output:0ParseExample/Const_6:output:0ParseExample/Const_7:output:0ParseExample/Const_8:output:0ParseExample/Const_9:output:0ParseExample/Const_10:output:0ParseExample/Const_11:output:0ParseExample/Const_12:output:0*
Tdense
2*
_output_shapesú
÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*`
dense_shapesP
N:::::::::::::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 x
transform_features_layer/ShapeShape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:v
,transform_features_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.transform_features_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transform_features_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&transform_features_layer/strided_sliceStridedSlice'transform_features_layer/Shape:output:05transform_features_layer/strided_slice/stack:output:07transform_features_layer/strided_slice/stack_1:output:07transform_features_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
 transform_features_layer/Shape_1Shape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:x
.transform_features_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0transform_features_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0transform_features_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(transform_features_layer/strided_slice_1StridedSlice)transform_features_layer/Shape_1:output:07transform_features_layer/strided_slice_1/stack:output:09transform_features_layer/strided_slice_1/stack_1:output:09transform_features_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'transform_features_layer/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ŕ
%transform_features_layer/zeros/packedPack1transform_features_layer/strided_slice_1:output:00transform_features_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
$transform_features_layer/zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ˇ
transform_features_layer/zerosFill.transform_features_layer/zeros/packed:output:0-transform_features_layer/zeros/Const:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ć
/transform_features_layer/PlaceholderWithDefaultPlaceholderWithDefault'transform_features_layer/zeros:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙

0transform_features_layer/StatefulPartitionedCallStatefulPartitionedCall*ParseExample/ParseExampleV2:dense_values:0*ParseExample/ParseExampleV2:dense_values:1*ParseExample/ParseExampleV2:dense_values:28transform_features_layer/PlaceholderWithDefault:output:0*ParseExample/ParseExampleV2:dense_values:3*ParseExample/ParseExampleV2:dense_values:4*ParseExample/ParseExampleV2:dense_values:5*ParseExample/ParseExampleV2:dense_values:6*ParseExample/ParseExampleV2:dense_values:7*ParseExample/ParseExampleV2:dense_values:8*ParseExample/ParseExampleV2:dense_values:9+ParseExample/ParseExampleV2:dense_values:10+ParseExample/ParseExampleV2:dense_values:11+ParseExample/ParseExampleV2:dense_values:12transform_features_layer_304558transform_features_layer_304560transform_features_layer_304562transform_features_layer_304564*
Tin
2	*
Tout
2											*
_output_shapesú
÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *"
fR
__inference_pruned_304423
model_4/CastCast9transform_features_layer/StatefulPartitionedCall:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_4/Cast_1Cast9transform_features_layer/StatefulPartitionedCall:output:1*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_4/Cast_2Cast9transform_features_layer/StatefulPartitionedCall:output:3*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_4/Cast_3Cast9transform_features_layer/StatefulPartitionedCall:output:4*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_4/Cast_4Cast9transform_features_layer/StatefulPartitionedCall:output:5*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_4/Cast_5Cast9transform_features_layer/StatefulPartitionedCall:output:6*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_4/Cast_6Cast9transform_features_layer/StatefulPartitionedCall:output:7*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_4/Cast_7Cast:transform_features_layer/StatefulPartitionedCall:output:10*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_4/Cast_8Cast:transform_features_layer/StatefulPartitionedCall:output:11*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_4/Cast_9Cast:transform_features_layer/StatefulPartitionedCall:output:12*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!model_4/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ë
model_4/concatenate_4/concatConcatV2model_4/Cast:y:0model_4/Cast_1:y:0model_4/Cast_2:y:0model_4/Cast_3:y:0model_4/Cast_4:y:0model_4/Cast_5:y:0model_4/Cast_6:y:09transform_features_layer/StatefulPartitionedCall:output:89transform_features_layer/StatefulPartitionedCall:output:9model_4/Cast_7:y:0model_4/Cast_8:y:0model_4/Cast_9:y:0*model_4/concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
&model_4/dense_68/MatMul/ReadVariableOpReadVariableOp/model_4_dense_68_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0Ť
model_4/dense_68/MatMulMatMul%model_4/concatenate_4/concat:output:0.model_4/dense_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
'model_4/dense_68/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_68_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0Ş
model_4/dense_68/BiasAddBiasAdd!model_4/dense_68/MatMul:product:0/model_4/dense_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ s
model_4/dense_68/ReluRelu!model_4/dense_68/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
&model_4/dense_69/MatMul/ReadVariableOpReadVariableOp/model_4_dense_69_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0¨
model_4/dense_69/MatMulMatMul#model_4/dense_68/Relu:activations:0.model_4/dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'model_4/dense_69/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_69_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Š
model_4/dense_69/BiasAddBiasAdd!model_4/dense_69/MatMul:product:0/model_4/dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙x
model_4/dense_69/SoftmaxSoftmax!model_4/dense_69/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q
IdentityIdentity"model_4/dense_69/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp(^model_4/dense_68/BiasAdd/ReadVariableOp'^model_4/dense_68/MatMul/ReadVariableOp(^model_4/dense_69/BiasAdd/ReadVariableOp'^model_4/dense_69/MatMul/ReadVariableOp1^transform_features_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : : : : : : : 2R
'model_4/dense_68/BiasAdd/ReadVariableOp'model_4/dense_68/BiasAdd/ReadVariableOp2P
&model_4/dense_68/MatMul/ReadVariableOp&model_4/dense_68/MatMul/ReadVariableOp2R
'model_4/dense_69/BiasAdd/ReadVariableOp'model_4/dense_69/BiasAdd/ReadVariableOp2P
&model_4/dense_69/MatMul/ReadVariableOp&model_4/dense_69/MatMul/ReadVariableOp2d
0transform_features_layer/StatefulPartitionedCall0transform_features_layer/StatefulPartitionedCall:M I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¤

ö
D__inference_dense_69_layer_call_and_return_conditional_losses_305349

inputs1
matmul_readvariableop_resource:	 -
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Äl
ő
__inference_pruned_304423

inputs
inputs_1
inputs_2
inputs_3	
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13-
)scale_to_0_1_min_and_max_identity_2_input-
)scale_to_0_1_min_and_max_identity_3_input/
+scale_to_0_1_1_min_and_max_identity_2_input/
+scale_to_0_1_1_min_and_max_identity_3_input
identity	

identity_1	

identity_2	

identity_3	

identity_4	

identity_5	

identity_6	

identity_7	

identity_8

identity_9
identity_10	
identity_11	
identity_12	c
 scale_to_0_1_1/min_and_max/ShapeConst*
_output_shapes
: *
dtype0*
valueB e
"scale_to_0_1_1/min_and_max/Shape_1Const*
_output_shapes
: *
dtype0*
valueB y
/scale_to_0_1_1/min_and_max/assert_equal_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ş
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:¨
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*:
value1B/ B)x (scale_to_0_1_1/min_and_max/Shape:0) = Ş
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*<
value3B1 B+y (scale_to_0_1_1/min_and_max/Shape_1:0) = a
scale_to_0_1/min_and_max/ShapeConst*
_output_shapes
: *
dtype0*
valueB c
 scale_to_0_1/min_and_max/Shape_1Const*
_output_shapes
: *
dtype0*
valueB w
-scale_to_0_1/min_and_max/assert_equal_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¨
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:¤
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*8
value/B- B'x (scale_to_0_1/min_and_max/Shape:0) = Ś
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*:
value1B/ B)y (scale_to_0_1/min_and_max/Shape_1:0) = e
 scale_to_0_1/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    W
scale_to_0_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
scale_to_0_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    g
"scale_to_0_1_1/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
scale_to_0_1_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?[
scale_to_0_1_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Q
inputs_copyIdentityinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙c
CastCastinputs_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ł
/scale_to_0_1_1/min_and_max/assert_equal_1/EqualEqual)scale_to_0_1_1/min_and_max/Shape:output:0+scale_to_0_1_1/min_and_max/Shape_1:output:0*
T0*
_output_shapes
: ť
-scale_to_0_1_1/min_and_max/assert_equal_1/AllAll3scale_to_0_1_1/min_and_max/assert_equal_1/Equal:z:08scale_to_0_1_1/min_and_max/assert_equal_1/Const:output:0*
_output_shapes
: ­
-scale_to_0_1/min_and_max/assert_equal_1/EqualEqual'scale_to_0_1/min_and_max/Shape:output:0)scale_to_0_1/min_and_max/Shape_1:output:0*
T0*
_output_shapes
: ľ
+scale_to_0_1/min_and_max/assert_equal_1/AllAll1scale_to_0_1/min_and_max/assert_equal_1/Equal:z:06scale_to_0_1/min_and_max/assert_equal_1/Const:output:0*
_output_shapes
: ě
5scale_to_0_1/min_and_max/assert_equal_1/Assert/AssertAssert4scale_to_0_1/min_and_max/assert_equal_1/All:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_0:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_1:output:0'scale_to_0_1/min_and_max/Shape:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_3:output:0)scale_to_0_1/min_and_max/Shape_1:output:0*
T	
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ˛
7scale_to_0_1_1/min_and_max/assert_equal_1/Assert/AssertAssert6scale_to_0_1_1/min_and_max/assert_equal_1/All:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_0:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_1:output:0)scale_to_0_1_1/min_and_max/Shape:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_3:output:0+scale_to_0_1_1/min_and_max/Shape_1:output:06^scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert*
T	
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ŕ
NoOpNoOp6^scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert8^scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert*"
_acd_function_control_output(*&
 _has_manual_control_dependencies(*
_output_shapes
 W
IdentityIdentityCast:y:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
Cast_1Castinputs_1_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_1Identity
Cast_1:y:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_3_copyIdentityinputs_3*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g

Identity_2Identityinputs_3_copy:output:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_4_copyIdentityinputs_4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
Cast_2Castinputs_4_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_3Identity
Cast_2:y:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_5_copyIdentityinputs_5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
Cast_8Castinputs_5_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_4Identity
Cast_8:y:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_6_copyIdentityinputs_6*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
Cast_7Castinputs_6_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_5Identity
Cast_7:y:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_7_copyIdentityinputs_7*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
Cast_9Castinputs_7_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_6Identity
Cast_9:y:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_8_copyIdentityinputs_8*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
Cast_3Castinputs_8_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_7Identity
Cast_3:y:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_9_copyIdentityinputs_9*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
#scale_to_0_1/min_and_max/Identity_2Identity)scale_to_0_1_min_and_max_identity_2_input*
T0*
_output_shapes
: 
scale_to_0_1/min_and_max/sub_1Sub)scale_to_0_1/min_and_max/sub_1/x:output:0,scale_to_0_1/min_and_max/Identity_2:output:0*
T0*
_output_shapes
: 
scale_to_0_1/subSubinputs_9_copy:output:0"scale_to_0_1/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
scale_to_0_1/zeros_like	ZerosLikescale_to_0_1/sub:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
#scale_to_0_1/min_and_max/Identity_3Identity)scale_to_0_1_min_and_max_identity_3_input*
T0*
_output_shapes
: 
scale_to_0_1/LessLess"scale_to_0_1/min_and_max/sub_1:z:0,scale_to_0_1/min_and_max/Identity_3:output:0*
T0*
_output_shapes
: `
scale_to_0_1/CastCastscale_to_0_1/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: 
scale_to_0_1/addAddV2scale_to_0_1/zeros_like:y:0scale_to_0_1/Cast:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
scale_to_0_1/Cast_1Castscale_to_0_1/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
scale_to_0_1/sub_1Sub,scale_to_0_1/min_and_max/Identity_3:output:0"scale_to_0_1/min_and_max/sub_1:z:0*
T0*
_output_shapes
: 
scale_to_0_1/truedivRealDivscale_to_0_1/sub:z:0scale_to_0_1/sub_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
scale_to_0_1/SigmoidSigmoidinputs_9_copy:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
scale_to_0_1/SelectV2SelectV2scale_to_0_1/Cast_1:y:0scale_to_0_1/truediv:z:0scale_to_0_1/Sigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
scale_to_0_1/mulMulscale_to_0_1/SelectV2:output:0scale_to_0_1/mul/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
scale_to_0_1/add_1AddV2scale_to_0_1/mul:z:0scale_to_0_1/add_1/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g

Identity_8Identityscale_to_0_1/add_1:z:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙W
inputs_10_copyIdentity	inputs_10*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
%scale_to_0_1_1/min_and_max/Identity_2Identity+scale_to_0_1_1_min_and_max_identity_2_input*
T0*
_output_shapes
: Ľ
 scale_to_0_1_1/min_and_max/sub_1Sub+scale_to_0_1_1/min_and_max/sub_1/x:output:0.scale_to_0_1_1/min_and_max/Identity_2:output:0*
T0*
_output_shapes
: 
scale_to_0_1_1/subSubinputs_10_copy:output:0$scale_to_0_1_1/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
scale_to_0_1_1/zeros_like	ZerosLikescale_to_0_1_1/sub:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
%scale_to_0_1_1/min_and_max/Identity_3Identity+scale_to_0_1_1_min_and_max_identity_3_input*
T0*
_output_shapes
: 
scale_to_0_1_1/LessLess$scale_to_0_1_1/min_and_max/sub_1:z:0.scale_to_0_1_1/min_and_max/Identity_3:output:0*
T0*
_output_shapes
: d
scale_to_0_1_1/CastCastscale_to_0_1_1/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: 
scale_to_0_1_1/addAddV2scale_to_0_1_1/zeros_like:y:0scale_to_0_1_1/Cast:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙v
scale_to_0_1_1/Cast_1Castscale_to_0_1_1/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
scale_to_0_1_1/sub_1Sub.scale_to_0_1_1/min_and_max/Identity_3:output:0$scale_to_0_1_1/min_and_max/sub_1:z:0*
T0*
_output_shapes
: 
scale_to_0_1_1/truedivRealDivscale_to_0_1_1/sub:z:0scale_to_0_1_1/sub_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
scale_to_0_1_1/SigmoidSigmoidinputs_10_copy:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
scale_to_0_1_1/SelectV2SelectV2scale_to_0_1_1/Cast_1:y:0scale_to_0_1_1/truediv:z:0scale_to_0_1_1/Sigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
scale_to_0_1_1/mulMul scale_to_0_1_1/SelectV2:output:0scale_to_0_1_1/mul/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
scale_to_0_1_1/add_1AddV2scale_to_0_1_1/mul:z:0scale_to_0_1_1/add_1/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i

Identity_9Identityscale_to_0_1_1/add_1:z:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙W
inputs_11_copyIdentity	inputs_11*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
Cast_6Castinputs_11_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Identity_10Identity
Cast_6:y:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙W
inputs_12_copyIdentity	inputs_12*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
Cast_4Castinputs_12_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Identity_11Identity
Cast_4:y:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙W
inputs_13_copyIdentity	inputs_13*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
Cast_5Castinputs_13_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Identity_12Identity
Cast_5:y:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*§
_input_shapes
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-	)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-
)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ź
č
.__inference_concatenate_4_layer_call_fn_305292
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
identityą
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_304929`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ů
_input_shapesç
ä:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_8:Q	M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_9:R
N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_11
Ő&

$__inference_signature_wrapper_304469

inputs
inputs_1
	inputs_10
	inputs_11
	inputs_12
	inputs_13
inputs_2
inputs_3	
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
unknown
	unknown_0
	unknown_1
	unknown_2
identity	

identity_1	

identity_2	

identity_3	

identity_4	

identity_5	

identity_6	

identity_7	

identity_8

identity_9
identity_10	
identity_11	
identity_12	˘StatefulPartitionedCallŠ
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13unknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2											*
_output_shapesú
÷:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *"
fR
__inference_pruned_304423`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_9Identity StatefulPartitionedCall:output:9^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
Identity_10Identity!StatefulPartitionedCall:output:10^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
Identity_11Identity!StatefulPartitionedCall:output:11^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
Identity_12Identity!StatefulPartitionedCall:output:12^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*§
_input_shapes
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1:RN
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_11:RN
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_12:RN
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_13:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_4:Q	M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_5:Q
M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_8:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_9:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ł

÷
D__inference_dense_68_layer_call_and_return_conditional_losses_304942

inputs1
matmul_readvariableop_resource:	 .
biasadd_readvariableop_resource:	 
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ć
Ë
C__inference_model_4_layer_call_and_return_conditional_losses_305077

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11"
dense_68_305066:	 
dense_68_305068:	 "
dense_69_305071:	 
dense_69_305073:
identity˘ dense_68/StatefulPartitionedCall˘ dense_69/StatefulPartitionedCall˝
concatenate_4/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_304929
 dense_68/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_68_305066dense_68_305068*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_304942
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_305071dense_69_305073*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_304959x
IdentityIdentity)dense_69/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesď
ě:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : 2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:O	K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:O
K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Š
serving_default
9
examples-
serving_default_examples:0˙˙˙˙˙˙˙˙˙<
output_00
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:śÇ
ü
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer_with_weights-0
layer-13
layer_with_weights-1
layer-14
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	tft_layer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ľ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ť
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
ť
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
Ë
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
$6 _saved_model_loader_tracked_dict"
_tf_keras_model
<
&0
'1
.2
/3"
trackable_list_wrapper
<
&0
'1
.2
/3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ő
<trace_0
=trace_1
>trace_2
?trace_32ę
(__inference_model_4_layer_call_fn_304977
(__inference_model_4_layer_call_fn_305190
(__inference_model_4_layer_call_fn_305214
(__inference_model_4_layer_call_fn_305112ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z<trace_0z=trace_1z>trace_2z?trace_3
Á
@trace_0
Atrace_1
Btrace_2
Ctrace_32Ö
C__inference_model_4_layer_call_and_return_conditional_losses_305245
C__inference_model_4_layer_call_and_return_conditional_losses_305276
C__inference_model_4_layer_call_and_return_conditional_losses_305138
C__inference_model_4_layer_call_and_return_conditional_losses_305164ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z@trace_0zAtrace_1zBtrace_2zCtrace_3
­BŞ
!__inference__wrapped_model_304661avg_cars_at home(approx).1_xfcoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfstore_sales(in millions)_xfstore_sqft_xftotal_children_xfunit_sales(in millions)_xfvideo_store_xf"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 

D
_variables
E_iterations
F_learning_rate
G_index_dict
H
_momentums
I_velocities
J_update_step_xla"
experimentalOptimizer
,
Kserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ň
Qtrace_02Ő
.__inference_concatenate_4_layer_call_fn_305292˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zQtrace_0

Rtrace_02đ
I__inference_concatenate_4_layer_call_and_return_conditional_losses_305309˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zRtrace_0
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
í
Xtrace_02Đ
)__inference_dense_68_layer_call_fn_305318˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zXtrace_0

Ytrace_02ë
D__inference_dense_68_layer_call_and_return_conditional_losses_305329˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zYtrace_0
": 	 2dense_68/kernel
: 2dense_68/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
í
_trace_02Đ
)__inference_dense_69_layer_call_fn_305338˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z_trace_0

`trace_02ë
D__inference_dense_69_layer_call_and_return_conditional_losses_305349˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z`trace_0
": 	 2dense_69/kernel
:2dense_69/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
Ň
ftrace_0
gtrace_12
9__inference_transform_features_layer_layer_call_fn_304773
9__inference_transform_features_layer_layer_call_fn_305396˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zftrace_0zgtrace_1

htrace_0
itrace_12Ń
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_305459
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_304883˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zhtrace_0zitrace_1

j	_imported
k_wrapped_function
l_structured_inputs
m_structured_outputs
n_output_to_inputs_map"
trackable_dict_wrapper
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŰBŘ
(__inference_model_4_layer_call_fn_304977avg_cars_at home(approx).1_xfcoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfstore_sales(in millions)_xfstore_sqft_xftotal_children_xfunit_sales(in millions)_xfvideo_store_xf"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ŻBŹ
(__inference_model_4_layer_call_fn_305190$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xf"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ŻBŹ
(__inference_model_4_layer_call_fn_305214$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xf"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ŰBŘ
(__inference_model_4_layer_call_fn_305112avg_cars_at home(approx).1_xfcoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfstore_sales(in millions)_xfstore_sqft_xftotal_children_xfunit_sales(in millions)_xfvideo_store_xf"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ĘBÇ
C__inference_model_4_layer_call_and_return_conditional_losses_305245$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xf"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ĘBÇ
C__inference_model_4_layer_call_and_return_conditional_losses_305276$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xf"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
öBó
C__inference_model_4_layer_call_and_return_conditional_losses_305138avg_cars_at home(approx).1_xfcoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfstore_sales(in millions)_xfstore_sqft_xftotal_children_xfunit_sales(in millions)_xfvideo_store_xf"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
öBó
C__inference_model_4_layer_call_and_return_conditional_losses_305164avg_cars_at home(approx).1_xfcoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfstore_sales(in millions)_xfstore_sqft_xftotal_children_xfunit_sales(in millions)_xfvideo_store_xf"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
_
E0
q1
r2
s3
t4
u5
v6
w7
x8"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
<
q0
s1
u2
w3"
trackable_list_wrapper
<
r0
t1
v2
x3"
trackable_list_wrapper
ż2źš
Ž˛Ş
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 0
Ä
y	capture_0
z	capture_1
{	capture_2
|	capture_3BÉ
$__inference_signature_wrapper_304629examples"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zy	capture_0zz	capture_1z{	capture_2z|	capture_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÔBŃ
.__inference_concatenate_4_layer_call_fn_305292inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ďBě
I__inference_concatenate_4_layer_call_and_return_conditional_losses_305309inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_68_layer_call_fn_305318inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
řBő
D__inference_dense_68_layer_call_and_return_conditional_losses_305329inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_69_layer_call_fn_305338inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
řBő
D__inference_dense_69_layer_call_and_return_conditional_losses_305349inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Š
y	capture_0
z	capture_1
{	capture_2
|	capture_3BŽ
9__inference_transform_features_layer_layer_call_fn_304773avg_cars_at home(approx).1
coffee_barcostfloristlow_fatnum_children_at_homeprepared_food	salad_barstore_sales(in millions)
store_sqfttotal_childrenunit_sales(in millions)video_store"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zy	capture_0zz	capture_1z{	capture_2z|	capture_3

y	capture_0
z	capture_1
{	capture_2
|	capture_3B
9__inference_transform_features_layer_layer_call_fn_305396!inputs_avg_cars_at_home_approx__1inputs_coffee_barinputs_costinputs_floristinputs_low_fatinputs_num_children_at_homeinputs_prepared_foodinputs_salad_barinputs_store_sales_in_millions_inputs_store_sqftinputs_total_childreninputs_unit_sales_in_millions_inputs_video_store"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zy	capture_0zz	capture_1z{	capture_2z|	capture_3

y	capture_0
z	capture_1
{	capture_2
|	capture_3B¤
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_305459!inputs_avg_cars_at_home_approx__1inputs_coffee_barinputs_costinputs_floristinputs_low_fatinputs_num_children_at_homeinputs_prepared_foodinputs_salad_barinputs_store_sales_in_millions_inputs_store_sqftinputs_total_childreninputs_unit_sales_in_millions_inputs_video_store"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zy	capture_0zz	capture_1z{	capture_2z|	capture_3
Ä
y	capture_0
z	capture_1
{	capture_2
|	capture_3BÉ
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_304883avg_cars_at home(approx).1
coffee_barcostfloristlow_fatnum_children_at_homeprepared_food	salad_barstore_sales(in millions)
store_sqfttotal_childrenunit_sales(in millions)video_store"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zy	capture_0zz	capture_1z{	capture_2z|	capture_3
Ä
}created_variables
~	resources
trackable_objects
initializers
assets

signatures
$_self_saveable_object_factories
ktransform_fn"
_generic_user_object
Ś
y	capture_0
z	capture_1
{	capture_2
|	capture_3BŤ
__inference_pruned_304423inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13zy	capture_0zz	capture_1z{	capture_2z|	capture_3
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
':%	 2Adam/m/dense_68/kernel
':%	 2Adam/v/dense_68/kernel
!: 2Adam/m/dense_68/bias
!: 2Adam/v/dense_68/bias
':%	 2Adam/m/dense_69/kernel
':%	 2Adam/v/dense_69/kernel
 :2Adam/m/dense_69/bias
 :2Adam/v/dense_69/bias
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
-
serving_default"
signature_map
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
Ć
y	capture_0
z	capture_1
{	capture_2
|	capture_3BË
$__inference_signature_wrapper_304469inputsinputs_1	inputs_10	inputs_11	inputs_12	inputs_13inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zy	capture_0zz	capture_1z{	capture_2z|	capture_3
!__inference__wrapped_model_304661ů&'./ť˘ˇ
Ż˘Ť
¨Ş¤
X
avg_cars_at home(approx).1_xf74
avg_cars_at home(approx).1_xf˙˙˙˙˙˙˙˙˙
8
coffee_bar_xf'$
coffee_bar_xf˙˙˙˙˙˙˙˙˙
2

florist_xf$!

florist_xf˙˙˙˙˙˙˙˙˙
2

low_fat_xf$!

low_fat_xf˙˙˙˙˙˙˙˙˙
L
num_children_at_home_xf1.
num_children_at_home_xf˙˙˙˙˙˙˙˙˙
>
prepared_food_xf*'
prepared_food_xf˙˙˙˙˙˙˙˙˙
6
salad_bar_xf&#
salad_bar_xf˙˙˙˙˙˙˙˙˙
T
store_sales(in millions)_xf52
store_sales(in millions)_xf˙˙˙˙˙˙˙˙˙
8
store_sqft_xf'$
store_sqft_xf˙˙˙˙˙˙˙˙˙
@
total_children_xf+(
total_children_xf˙˙˙˙˙˙˙˙˙
R
unit_sales(in millions)_xf41
unit_sales(in millions)_xf˙˙˙˙˙˙˙˙˙
:
video_store_xf(%
video_store_xf˙˙˙˙˙˙˙˙˙
Ş "3Ş0
.
dense_69"
dense_69˙˙˙˙˙˙˙˙˙Č
I__inference_concatenate_4_layer_call_and_return_conditional_losses_305309úÉ˘Ĺ
˝˘š
ś˛
"
inputs_0˙˙˙˙˙˙˙˙˙
"
inputs_1˙˙˙˙˙˙˙˙˙
"
inputs_2˙˙˙˙˙˙˙˙˙
"
inputs_3˙˙˙˙˙˙˙˙˙
"
inputs_4˙˙˙˙˙˙˙˙˙
"
inputs_5˙˙˙˙˙˙˙˙˙
"
inputs_6˙˙˙˙˙˙˙˙˙
"
inputs_7˙˙˙˙˙˙˙˙˙
"
inputs_8˙˙˙˙˙˙˙˙˙
"
inputs_9˙˙˙˙˙˙˙˙˙
# 
	inputs_10˙˙˙˙˙˙˙˙˙
# 
	inputs_11˙˙˙˙˙˙˙˙˙
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 ˘
.__inference_concatenate_4_layer_call_fn_305292ďÉ˘Ĺ
˝˘š
ś˛
"
inputs_0˙˙˙˙˙˙˙˙˙
"
inputs_1˙˙˙˙˙˙˙˙˙
"
inputs_2˙˙˙˙˙˙˙˙˙
"
inputs_3˙˙˙˙˙˙˙˙˙
"
inputs_4˙˙˙˙˙˙˙˙˙
"
inputs_5˙˙˙˙˙˙˙˙˙
"
inputs_6˙˙˙˙˙˙˙˙˙
"
inputs_7˙˙˙˙˙˙˙˙˙
"
inputs_8˙˙˙˙˙˙˙˙˙
"
inputs_9˙˙˙˙˙˙˙˙˙
# 
	inputs_10˙˙˙˙˙˙˙˙˙
# 
	inputs_11˙˙˙˙˙˙˙˙˙
Ş "!
unknown˙˙˙˙˙˙˙˙˙Ź
D__inference_dense_68_layer_call_and_return_conditional_losses_305329d&'/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "-˘*
# 
tensor_0˙˙˙˙˙˙˙˙˙ 
 
)__inference_dense_68_layer_call_fn_305318Y&'/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş ""
unknown˙˙˙˙˙˙˙˙˙ Ź
D__inference_dense_69_layer_call_and_return_conditional_losses_305349d./0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙ 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 
)__inference_dense_69_layer_call_fn_305338Y./0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙ 
Ş "!
unknown˙˙˙˙˙˙˙˙˙Â
C__inference_model_4_layer_call_and_return_conditional_losses_305138ú&'./Ă˘ż
ˇ˘ł
¨Ş¤
X
avg_cars_at home(approx).1_xf74
avg_cars_at home(approx).1_xf˙˙˙˙˙˙˙˙˙
8
coffee_bar_xf'$
coffee_bar_xf˙˙˙˙˙˙˙˙˙
2

florist_xf$!

florist_xf˙˙˙˙˙˙˙˙˙
2

low_fat_xf$!

low_fat_xf˙˙˙˙˙˙˙˙˙
L
num_children_at_home_xf1.
num_children_at_home_xf˙˙˙˙˙˙˙˙˙
>
prepared_food_xf*'
prepared_food_xf˙˙˙˙˙˙˙˙˙
6
salad_bar_xf&#
salad_bar_xf˙˙˙˙˙˙˙˙˙
T
store_sales(in millions)_xf52
store_sales(in millions)_xf˙˙˙˙˙˙˙˙˙
8
store_sqft_xf'$
store_sqft_xf˙˙˙˙˙˙˙˙˙
@
total_children_xf+(
total_children_xf˙˙˙˙˙˙˙˙˙
R
unit_sales(in millions)_xf41
unit_sales(in millions)_xf˙˙˙˙˙˙˙˙˙
:
video_store_xf(%
video_store_xf˙˙˙˙˙˙˙˙˙
p 

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 Â
C__inference_model_4_layer_call_and_return_conditional_losses_305164ú&'./Ă˘ż
ˇ˘ł
¨Ş¤
X
avg_cars_at home(approx).1_xf74
avg_cars_at home(approx).1_xf˙˙˙˙˙˙˙˙˙
8
coffee_bar_xf'$
coffee_bar_xf˙˙˙˙˙˙˙˙˙
2

florist_xf$!

florist_xf˙˙˙˙˙˙˙˙˙
2

low_fat_xf$!

low_fat_xf˙˙˙˙˙˙˙˙˙
L
num_children_at_home_xf1.
num_children_at_home_xf˙˙˙˙˙˙˙˙˙
>
prepared_food_xf*'
prepared_food_xf˙˙˙˙˙˙˙˙˙
6
salad_bar_xf&#
salad_bar_xf˙˙˙˙˙˙˙˙˙
T
store_sales(in millions)_xf52
store_sales(in millions)_xf˙˙˙˙˙˙˙˙˙
8
store_sqft_xf'$
store_sqft_xf˙˙˙˙˙˙˙˙˙
@
total_children_xf+(
total_children_xf˙˙˙˙˙˙˙˙˙
R
unit_sales(in millions)_xf41
unit_sales(in millions)_xf˙˙˙˙˙˙˙˙˙
:
video_store_xf(%
video_store_xf˙˙˙˙˙˙˙˙˙
p

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 
C__inference_model_4_layer_call_and_return_conditional_losses_305245Î&'./˘
˘
üŞř
_
avg_cars_at home(approx).1_xf>;
$inputs_avg_cars_at_home_approx__1_xf˙˙˙˙˙˙˙˙˙
?
coffee_bar_xf.+
inputs_coffee_bar_xf˙˙˙˙˙˙˙˙˙
9

florist_xf+(
inputs_florist_xf˙˙˙˙˙˙˙˙˙
9

low_fat_xf+(
inputs_low_fat_xf˙˙˙˙˙˙˙˙˙
S
num_children_at_home_xf85
inputs_num_children_at_home_xf˙˙˙˙˙˙˙˙˙
E
prepared_food_xf1.
inputs_prepared_food_xf˙˙˙˙˙˙˙˙˙
=
salad_bar_xf-*
inputs_salad_bar_xf˙˙˙˙˙˙˙˙˙
[
store_sales(in millions)_xf<9
"inputs_store_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
?
store_sqft_xf.+
inputs_store_sqft_xf˙˙˙˙˙˙˙˙˙
G
total_children_xf2/
inputs_total_children_xf˙˙˙˙˙˙˙˙˙
Y
unit_sales(in millions)_xf;8
!inputs_unit_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
A
video_store_xf/,
inputs_video_store_xf˙˙˙˙˙˙˙˙˙
p 

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 
C__inference_model_4_layer_call_and_return_conditional_losses_305276Î&'./˘
˘
üŞř
_
avg_cars_at home(approx).1_xf>;
$inputs_avg_cars_at_home_approx__1_xf˙˙˙˙˙˙˙˙˙
?
coffee_bar_xf.+
inputs_coffee_bar_xf˙˙˙˙˙˙˙˙˙
9

florist_xf+(
inputs_florist_xf˙˙˙˙˙˙˙˙˙
9

low_fat_xf+(
inputs_low_fat_xf˙˙˙˙˙˙˙˙˙
S
num_children_at_home_xf85
inputs_num_children_at_home_xf˙˙˙˙˙˙˙˙˙
E
prepared_food_xf1.
inputs_prepared_food_xf˙˙˙˙˙˙˙˙˙
=
salad_bar_xf-*
inputs_salad_bar_xf˙˙˙˙˙˙˙˙˙
[
store_sales(in millions)_xf<9
"inputs_store_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
?
store_sqft_xf.+
inputs_store_sqft_xf˙˙˙˙˙˙˙˙˙
G
total_children_xf2/
inputs_total_children_xf˙˙˙˙˙˙˙˙˙
Y
unit_sales(in millions)_xf;8
!inputs_unit_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
A
video_store_xf/,
inputs_video_store_xf˙˙˙˙˙˙˙˙˙
p

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 
(__inference_model_4_layer_call_fn_304977ď&'./Ă˘ż
ˇ˘ł
¨Ş¤
X
avg_cars_at home(approx).1_xf74
avg_cars_at home(approx).1_xf˙˙˙˙˙˙˙˙˙
8
coffee_bar_xf'$
coffee_bar_xf˙˙˙˙˙˙˙˙˙
2

florist_xf$!

florist_xf˙˙˙˙˙˙˙˙˙
2

low_fat_xf$!

low_fat_xf˙˙˙˙˙˙˙˙˙
L
num_children_at_home_xf1.
num_children_at_home_xf˙˙˙˙˙˙˙˙˙
>
prepared_food_xf*'
prepared_food_xf˙˙˙˙˙˙˙˙˙
6
salad_bar_xf&#
salad_bar_xf˙˙˙˙˙˙˙˙˙
T
store_sales(in millions)_xf52
store_sales(in millions)_xf˙˙˙˙˙˙˙˙˙
8
store_sqft_xf'$
store_sqft_xf˙˙˙˙˙˙˙˙˙
@
total_children_xf+(
total_children_xf˙˙˙˙˙˙˙˙˙
R
unit_sales(in millions)_xf41
unit_sales(in millions)_xf˙˙˙˙˙˙˙˙˙
:
video_store_xf(%
video_store_xf˙˙˙˙˙˙˙˙˙
p 

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
(__inference_model_4_layer_call_fn_305112ď&'./Ă˘ż
ˇ˘ł
¨Ş¤
X
avg_cars_at home(approx).1_xf74
avg_cars_at home(approx).1_xf˙˙˙˙˙˙˙˙˙
8
coffee_bar_xf'$
coffee_bar_xf˙˙˙˙˙˙˙˙˙
2

florist_xf$!

florist_xf˙˙˙˙˙˙˙˙˙
2

low_fat_xf$!

low_fat_xf˙˙˙˙˙˙˙˙˙
L
num_children_at_home_xf1.
num_children_at_home_xf˙˙˙˙˙˙˙˙˙
>
prepared_food_xf*'
prepared_food_xf˙˙˙˙˙˙˙˙˙
6
salad_bar_xf&#
salad_bar_xf˙˙˙˙˙˙˙˙˙
T
store_sales(in millions)_xf52
store_sales(in millions)_xf˙˙˙˙˙˙˙˙˙
8
store_sqft_xf'$
store_sqft_xf˙˙˙˙˙˙˙˙˙
@
total_children_xf+(
total_children_xf˙˙˙˙˙˙˙˙˙
R
unit_sales(in millions)_xf41
unit_sales(in millions)_xf˙˙˙˙˙˙˙˙˙
:
video_store_xf(%
video_store_xf˙˙˙˙˙˙˙˙˙
p

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙đ
(__inference_model_4_layer_call_fn_305190Ă&'./˘
˘
üŞř
_
avg_cars_at home(approx).1_xf>;
$inputs_avg_cars_at_home_approx__1_xf˙˙˙˙˙˙˙˙˙
?
coffee_bar_xf.+
inputs_coffee_bar_xf˙˙˙˙˙˙˙˙˙
9

florist_xf+(
inputs_florist_xf˙˙˙˙˙˙˙˙˙
9

low_fat_xf+(
inputs_low_fat_xf˙˙˙˙˙˙˙˙˙
S
num_children_at_home_xf85
inputs_num_children_at_home_xf˙˙˙˙˙˙˙˙˙
E
prepared_food_xf1.
inputs_prepared_food_xf˙˙˙˙˙˙˙˙˙
=
salad_bar_xf-*
inputs_salad_bar_xf˙˙˙˙˙˙˙˙˙
[
store_sales(in millions)_xf<9
"inputs_store_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
?
store_sqft_xf.+
inputs_store_sqft_xf˙˙˙˙˙˙˙˙˙
G
total_children_xf2/
inputs_total_children_xf˙˙˙˙˙˙˙˙˙
Y
unit_sales(in millions)_xf;8
!inputs_unit_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
A
video_store_xf/,
inputs_video_store_xf˙˙˙˙˙˙˙˙˙
p 

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙đ
(__inference_model_4_layer_call_fn_305214Ă&'./˘
˘
üŞř
_
avg_cars_at home(approx).1_xf>;
$inputs_avg_cars_at_home_approx__1_xf˙˙˙˙˙˙˙˙˙
?
coffee_bar_xf.+
inputs_coffee_bar_xf˙˙˙˙˙˙˙˙˙
9

florist_xf+(
inputs_florist_xf˙˙˙˙˙˙˙˙˙
9

low_fat_xf+(
inputs_low_fat_xf˙˙˙˙˙˙˙˙˙
S
num_children_at_home_xf85
inputs_num_children_at_home_xf˙˙˙˙˙˙˙˙˙
E
prepared_food_xf1.
inputs_prepared_food_xf˙˙˙˙˙˙˙˙˙
=
salad_bar_xf-*
inputs_salad_bar_xf˙˙˙˙˙˙˙˙˙
[
store_sales(in millions)_xf<9
"inputs_store_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
?
store_sqft_xf.+
inputs_store_sqft_xf˙˙˙˙˙˙˙˙˙
G
total_children_xf2/
inputs_total_children_xf˙˙˙˙˙˙˙˙˙
Y
unit_sales(in millions)_xf;8
!inputs_unit_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
A
video_store_xf/,
inputs_video_store_xf˙˙˙˙˙˙˙˙˙
p

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙ľ
__inference_pruned_304423yz{|­˘Š
Ą˘
Ş
Y
avg_cars_at home(approx).1;8
!inputs_avg_cars_at_home_approx__1˙˙˙˙˙˙˙˙˙
9

coffee_bar+(
inputs_coffee_bar˙˙˙˙˙˙˙˙˙
-
cost%"
inputs_cost˙˙˙˙˙˙˙˙˙
5
cost_bin)&
inputs_cost_bin˙˙˙˙˙˙˙˙˙	
3
florist(%
inputs_florist˙˙˙˙˙˙˙˙˙
3
low_fat(%
inputs_low_fat˙˙˙˙˙˙˙˙˙
M
num_children_at_home52
inputs_num_children_at_home˙˙˙˙˙˙˙˙˙
?
prepared_food.+
inputs_prepared_food˙˙˙˙˙˙˙˙˙
7
	salad_bar*'
inputs_salad_bar˙˙˙˙˙˙˙˙˙
U
store_sales(in millions)96
inputs_store_sales_in_millions_˙˙˙˙˙˙˙˙˙
9

store_sqft+(
inputs_store_sqft˙˙˙˙˙˙˙˙˙
A
total_children/,
inputs_total_children˙˙˙˙˙˙˙˙˙
S
unit_sales(in millions)85
inputs_unit_sales_in_millions_˙˙˙˙˙˙˙˙˙
;
video_store,)
inputs_video_store˙˙˙˙˙˙˙˙˙
Ş "ŢŞÚ
X
avg_cars_at home(approx).1_xf74
avg_cars_at_home_approx__1_xf˙˙˙˙˙˙˙˙˙	
8
coffee_bar_xf'$
coffee_bar_xf˙˙˙˙˙˙˙˙˙	
4
cost_bin_xf%"
cost_bin_xf˙˙˙˙˙˙˙˙˙	
2

florist_xf$!

florist_xf˙˙˙˙˙˙˙˙˙	
2

low_fat_xf$!

low_fat_xf˙˙˙˙˙˙˙˙˙	
L
num_children_at_home_xf1.
num_children_at_home_xf˙˙˙˙˙˙˙˙˙	
>
prepared_food_xf*'
prepared_food_xf˙˙˙˙˙˙˙˙˙	
6
salad_bar_xf&#
salad_bar_xf˙˙˙˙˙˙˙˙˙	
T
store_sales(in millions)_xf52
store_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
8
store_sqft_xf'$
store_sqft_xf˙˙˙˙˙˙˙˙˙
@
total_children_xf+(
total_children_xf˙˙˙˙˙˙˙˙˙	
R
unit_sales(in millions)_xf41
unit_sales_in_millions__xf˙˙˙˙˙˙˙˙˙	
:
video_store_xf(%
video_store_xf˙˙˙˙˙˙˙˙˙	Ç
$__inference_signature_wrapper_304469yz{|´˘°
˘ 
¨Ş¤
*
inputs 
inputs˙˙˙˙˙˙˙˙˙
.
inputs_1"
inputs_1˙˙˙˙˙˙˙˙˙
0
	inputs_10# 
	inputs_10˙˙˙˙˙˙˙˙˙
0
	inputs_11# 
	inputs_11˙˙˙˙˙˙˙˙˙
0
	inputs_12# 
	inputs_12˙˙˙˙˙˙˙˙˙
0
	inputs_13# 
	inputs_13˙˙˙˙˙˙˙˙˙
.
inputs_2"
inputs_2˙˙˙˙˙˙˙˙˙
.
inputs_3"
inputs_3˙˙˙˙˙˙˙˙˙	
.
inputs_4"
inputs_4˙˙˙˙˙˙˙˙˙
.
inputs_5"
inputs_5˙˙˙˙˙˙˙˙˙
.
inputs_6"
inputs_6˙˙˙˙˙˙˙˙˙
.
inputs_7"
inputs_7˙˙˙˙˙˙˙˙˙
.
inputs_8"
inputs_8˙˙˙˙˙˙˙˙˙
.
inputs_9"
inputs_9˙˙˙˙˙˙˙˙˙"ŢŞÚ
X
avg_cars_at home(approx).1_xf74
avg_cars_at_home_approx__1_xf˙˙˙˙˙˙˙˙˙	
8
coffee_bar_xf'$
coffee_bar_xf˙˙˙˙˙˙˙˙˙	
4
cost_bin_xf%"
cost_bin_xf˙˙˙˙˙˙˙˙˙	
2

florist_xf$!

florist_xf˙˙˙˙˙˙˙˙˙	
2

low_fat_xf$!

low_fat_xf˙˙˙˙˙˙˙˙˙	
L
num_children_at_home_xf1.
num_children_at_home_xf˙˙˙˙˙˙˙˙˙	
>
prepared_food_xf*'
prepared_food_xf˙˙˙˙˙˙˙˙˙	
6
salad_bar_xf&#
salad_bar_xf˙˙˙˙˙˙˙˙˙	
T
store_sales(in millions)_xf52
store_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
8
store_sqft_xf'$
store_sqft_xf˙˙˙˙˙˙˙˙˙
@
total_children_xf+(
total_children_xf˙˙˙˙˙˙˙˙˙	
R
unit_sales(in millions)_xf41
unit_sales_in_millions__xf˙˙˙˙˙˙˙˙˙	
:
video_store_xf(%
video_store_xf˙˙˙˙˙˙˙˙˙	˘
$__inference_signature_wrapper_304629zyz{|&'./9˘6
˘ 
/Ş,
*
examples
examples˙˙˙˙˙˙˙˙˙"3Ş0
.
output_0"
output_0˙˙˙˙˙˙˙˙˙ 
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_304883Çyz{|˘
˘
Ş
R
avg_cars_at home(approx).141
avg_cars_at home(approx).1˙˙˙˙˙˙˙˙˙
2

coffee_bar$!

coffee_bar˙˙˙˙˙˙˙˙˙
&
cost
cost˙˙˙˙˙˙˙˙˙
,
florist!
florist˙˙˙˙˙˙˙˙˙
,
low_fat!
low_fat˙˙˙˙˙˙˙˙˙
F
num_children_at_home.+
num_children_at_home˙˙˙˙˙˙˙˙˙
8
prepared_food'$
prepared_food˙˙˙˙˙˙˙˙˙
0
	salad_bar# 
	salad_bar˙˙˙˙˙˙˙˙˙
N
store_sales(in millions)2/
store_sales(in millions)˙˙˙˙˙˙˙˙˙
2

store_sqft$!

store_sqft˙˙˙˙˙˙˙˙˙
:
total_children(%
total_children˙˙˙˙˙˙˙˙˙
L
unit_sales(in millions)1.
unit_sales(in millions)˙˙˙˙˙˙˙˙˙
4
video_store%"
video_store˙˙˙˙˙˙˙˙˙
Ş " ˘
Ş
a
avg_cars_at home(approx).1_xf@=
&tensor_0_avg_cars_at_home_approx__1_xf˙˙˙˙˙˙˙˙˙	
A
coffee_bar_xf0-
tensor_0_coffee_bar_xf˙˙˙˙˙˙˙˙˙	
;

florist_xf-*
tensor_0_florist_xf˙˙˙˙˙˙˙˙˙	
;

low_fat_xf-*
tensor_0_low_fat_xf˙˙˙˙˙˙˙˙˙	
U
num_children_at_home_xf:7
 tensor_0_num_children_at_home_xf˙˙˙˙˙˙˙˙˙	
G
prepared_food_xf30
tensor_0_prepared_food_xf˙˙˙˙˙˙˙˙˙	
?
salad_bar_xf/,
tensor_0_salad_bar_xf˙˙˙˙˙˙˙˙˙	
]
store_sales(in millions)_xf>;
$tensor_0_store_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
A
store_sqft_xf0-
tensor_0_store_sqft_xf˙˙˙˙˙˙˙˙˙
I
total_children_xf41
tensor_0_total_children_xf˙˙˙˙˙˙˙˙˙	
[
unit_sales(in millions)_xf=:
#tensor_0_unit_sales_in_millions__xf˙˙˙˙˙˙˙˙˙	
C
video_store_xf1.
tensor_0_video_store_xf˙˙˙˙˙˙˙˙˙	
 ű
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_305459˘yz{|ö˘ň
ę˘ć
ăŞß
Y
avg_cars_at home(approx).1;8
!inputs_avg_cars_at_home_approx__1˙˙˙˙˙˙˙˙˙
9

coffee_bar+(
inputs_coffee_bar˙˙˙˙˙˙˙˙˙
-
cost%"
inputs_cost˙˙˙˙˙˙˙˙˙
3
florist(%
inputs_florist˙˙˙˙˙˙˙˙˙
3
low_fat(%
inputs_low_fat˙˙˙˙˙˙˙˙˙
M
num_children_at_home52
inputs_num_children_at_home˙˙˙˙˙˙˙˙˙
?
prepared_food.+
inputs_prepared_food˙˙˙˙˙˙˙˙˙
7
	salad_bar*'
inputs_salad_bar˙˙˙˙˙˙˙˙˙
U
store_sales(in millions)96
inputs_store_sales_in_millions_˙˙˙˙˙˙˙˙˙
9

store_sqft+(
inputs_store_sqft˙˙˙˙˙˙˙˙˙
A
total_children/,
inputs_total_children˙˙˙˙˙˙˙˙˙
S
unit_sales(in millions)85
inputs_unit_sales_in_millions_˙˙˙˙˙˙˙˙˙
;
video_store,)
inputs_video_store˙˙˙˙˙˙˙˙˙
Ş " ˘
Ş
a
avg_cars_at home(approx).1_xf@=
&tensor_0_avg_cars_at_home_approx__1_xf˙˙˙˙˙˙˙˙˙	
A
coffee_bar_xf0-
tensor_0_coffee_bar_xf˙˙˙˙˙˙˙˙˙	
;

florist_xf-*
tensor_0_florist_xf˙˙˙˙˙˙˙˙˙	
;

low_fat_xf-*
tensor_0_low_fat_xf˙˙˙˙˙˙˙˙˙	
U
num_children_at_home_xf:7
 tensor_0_num_children_at_home_xf˙˙˙˙˙˙˙˙˙	
G
prepared_food_xf30
tensor_0_prepared_food_xf˙˙˙˙˙˙˙˙˙	
?
salad_bar_xf/,
tensor_0_salad_bar_xf˙˙˙˙˙˙˙˙˙	
]
store_sales(in millions)_xf>;
$tensor_0_store_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
A
store_sqft_xf0-
tensor_0_store_sqft_xf˙˙˙˙˙˙˙˙˙
I
total_children_xf41
tensor_0_total_children_xf˙˙˙˙˙˙˙˙˙	
[
unit_sales(in millions)_xf=:
#tensor_0_unit_sales_in_millions__xf˙˙˙˙˙˙˙˙˙	
C
video_store_xf1.
tensor_0_video_store_xf˙˙˙˙˙˙˙˙˙	
 
9__inference_transform_features_layer_layer_call_fn_304773Ďyz{|˘
˘
Ş
R
avg_cars_at home(approx).141
avg_cars_at home(approx).1˙˙˙˙˙˙˙˙˙
2

coffee_bar$!

coffee_bar˙˙˙˙˙˙˙˙˙
&
cost
cost˙˙˙˙˙˙˙˙˙
,
florist!
florist˙˙˙˙˙˙˙˙˙
,
low_fat!
low_fat˙˙˙˙˙˙˙˙˙
F
num_children_at_home.+
num_children_at_home˙˙˙˙˙˙˙˙˙
8
prepared_food'$
prepared_food˙˙˙˙˙˙˙˙˙
0
	salad_bar# 
	salad_bar˙˙˙˙˙˙˙˙˙
N
store_sales(in millions)2/
store_sales(in millions)˙˙˙˙˙˙˙˙˙
2

store_sqft$!

store_sqft˙˙˙˙˙˙˙˙˙
:
total_children(%
total_children˙˙˙˙˙˙˙˙˙
L
unit_sales(in millions)1.
unit_sales(in millions)˙˙˙˙˙˙˙˙˙
4
video_store%"
video_store˙˙˙˙˙˙˙˙˙
Ş "¨Ş¤
X
avg_cars_at home(approx).1_xf74
avg_cars_at_home_approx__1_xf˙˙˙˙˙˙˙˙˙	
8
coffee_bar_xf'$
coffee_bar_xf˙˙˙˙˙˙˙˙˙	
2

florist_xf$!

florist_xf˙˙˙˙˙˙˙˙˙	
2

low_fat_xf$!

low_fat_xf˙˙˙˙˙˙˙˙˙	
L
num_children_at_home_xf1.
num_children_at_home_xf˙˙˙˙˙˙˙˙˙	
>
prepared_food_xf*'
prepared_food_xf˙˙˙˙˙˙˙˙˙	
6
salad_bar_xf&#
salad_bar_xf˙˙˙˙˙˙˙˙˙	
T
store_sales(in millions)_xf52
store_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
8
store_sqft_xf'$
store_sqft_xf˙˙˙˙˙˙˙˙˙
@
total_children_xf+(
total_children_xf˙˙˙˙˙˙˙˙˙	
R
unit_sales(in millions)_xf41
unit_sales_in_millions__xf˙˙˙˙˙˙˙˙˙	
:
video_store_xf(%
video_store_xf˙˙˙˙˙˙˙˙˙	č
9__inference_transform_features_layer_layer_call_fn_305396Şyz{|ö˘ň
ę˘ć
ăŞß
Y
avg_cars_at home(approx).1;8
!inputs_avg_cars_at_home_approx__1˙˙˙˙˙˙˙˙˙
9

coffee_bar+(
inputs_coffee_bar˙˙˙˙˙˙˙˙˙
-
cost%"
inputs_cost˙˙˙˙˙˙˙˙˙
3
florist(%
inputs_florist˙˙˙˙˙˙˙˙˙
3
low_fat(%
inputs_low_fat˙˙˙˙˙˙˙˙˙
M
num_children_at_home52
inputs_num_children_at_home˙˙˙˙˙˙˙˙˙
?
prepared_food.+
inputs_prepared_food˙˙˙˙˙˙˙˙˙
7
	salad_bar*'
inputs_salad_bar˙˙˙˙˙˙˙˙˙
U
store_sales(in millions)96
inputs_store_sales_in_millions_˙˙˙˙˙˙˙˙˙
9

store_sqft+(
inputs_store_sqft˙˙˙˙˙˙˙˙˙
A
total_children/,
inputs_total_children˙˙˙˙˙˙˙˙˙
S
unit_sales(in millions)85
inputs_unit_sales_in_millions_˙˙˙˙˙˙˙˙˙
;
video_store,)
inputs_video_store˙˙˙˙˙˙˙˙˙
Ş "¨Ş¤
X
avg_cars_at home(approx).1_xf74
avg_cars_at_home_approx__1_xf˙˙˙˙˙˙˙˙˙	
8
coffee_bar_xf'$
coffee_bar_xf˙˙˙˙˙˙˙˙˙	
2

florist_xf$!

florist_xf˙˙˙˙˙˙˙˙˙	
2

low_fat_xf$!

low_fat_xf˙˙˙˙˙˙˙˙˙	
L
num_children_at_home_xf1.
num_children_at_home_xf˙˙˙˙˙˙˙˙˙	
>
prepared_food_xf*'
prepared_food_xf˙˙˙˙˙˙˙˙˙	
6
salad_bar_xf&#
salad_bar_xf˙˙˙˙˙˙˙˙˙	
T
store_sales(in millions)_xf52
store_sales_in_millions__xf˙˙˙˙˙˙˙˙˙
8
store_sqft_xf'$
store_sqft_xf˙˙˙˙˙˙˙˙˙
@
total_children_xf+(
total_children_xf˙˙˙˙˙˙˙˙˙	
R
unit_sales(in millions)_xf41
unit_sales_in_millions__xf˙˙˙˙˙˙˙˙˙	
:
video_store_xf(%
video_store_xf˙˙˙˙˙˙˙˙˙	