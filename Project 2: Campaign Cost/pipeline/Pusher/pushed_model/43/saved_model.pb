�
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
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
	summarizeint�
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
incompatible_shape_errorbool(�
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.11.12v2.11.0-94-ga3e2c692c188݆
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
 * ���
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *)\�A
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *\��
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
�
Adam/v/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_23/bias
y
(Adam/v/dense_23/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_23/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_23/bias
y
(Adam/m/dense_23/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_23/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_23/kernel
�
*Adam/v/dense_23/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_23/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_23/kernel
�
*Adam/m/dense_23/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_23/kernel*
_output_shapes

:*
dtype0
�
Adam/v/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_22/bias
y
(Adam/v/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_22/bias
y
(Adam/m/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_22/kernel
�
*Adam/v/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_22/kernel
�
*Adam/m/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/kernel*
_output_shapes

:*
dtype0
�
Adam/v/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_21/bias
y
(Adam/v/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_21/bias
y
(Adam/m/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/v/dense_21/kernel
�
*Adam/v/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/m/dense_21/kernel
�
*Adam/m/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/dense_20/bias
z
(Adam/v/dense_20/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_20/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/dense_20/bias
z
(Adam/m/dense_20/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_20/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/v/dense_20/kernel
�
*Adam/v/dense_20/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_20/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/m/dense_20/kernel
�
*Adam/m/dense_20/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_20/kernel*
_output_shapes
:	�*
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
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:*
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:*
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:*
dtype0
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:*
dtype0
{
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	�*
dtype0
s
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_20/bias
l
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes	
:�*
dtype0
{
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_20/kernel
t
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes
:	�*
dtype0
s
serving_default_examplesPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_examplesConst_3Const_2Const_1Constdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_257033

NoOpNoOp
�G
Const_4Const"/device:CPU:0*
_output_shapes
: *
dtype0*�F
value�FB�F B�F
�
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
layer_with_weights-2
layer-15
layer_with_weights-3
layer-16
layer-17
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	tft_layer

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
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses* 
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias*
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

0kernel
1bias*
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias*
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
$H _saved_model_loader_tracked_dict* 
<
(0
)1
02
13
84
95
@6
A7*
<
(0
)1
02
13
84
95
@6
A7*
* 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ntrace_0
Otrace_1
Ptrace_2
Qtrace_3* 
6
Rtrace_0
Strace_1
Ttrace_2
Utrace_3* 
* 
�
V
_variables
W_iterations
X_learning_rate
Y_index_dict
Z
_momentums
[_velocities
\_update_step_xla*

]serving_default* 
* 
* 
* 
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 

ctrace_0* 

dtrace_0* 

(0
)1*

(0
)1*
* 
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

jtrace_0* 

ktrace_0* 
_Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_20/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

00
11*

00
11*
* 
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

qtrace_0* 

rtrace_0* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

xtrace_0* 

ytrace_0* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

@0
A1*
* 
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
y
�	_imported
�_wrapped_function
�_structured_inputs
�_structured_outputs
�_output_to_inputs_map* 
* 
�
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
15
16
17*

�0
�1*
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
�
W0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
B
�	capture_0
�	capture_1
�	capture_2
�	capture_3* 
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
B
�	capture_0
�	capture_1
�	capture_2
�	capture_3* 
B
�	capture_0
�	capture_1
�	capture_2
�	capture_3* 
B
�	capture_0
�	capture_1
�	capture_2
�	capture_3* 
B
�	capture_0
�	capture_1
�	capture_2
�	capture_3* 
�
�created_variables
�	resources
�trackable_objects
�initializers
�assets
�
signatures
$�_self_saveable_object_factories
�transform_fn* 
B
�	capture_0
�	capture_1
�	capture_2
�	capture_3* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/dense_20/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_20/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_20/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_20/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_21/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_21/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_21/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_21/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_22/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_22/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_22/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_22/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_23/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_23/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_23/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_23/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
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
�serving_default* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
B
�	capture_0
�	capture_1
�	capture_2
�	capture_3* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp*Adam/m/dense_20/kernel/Read/ReadVariableOp*Adam/v/dense_20/kernel/Read/ReadVariableOp(Adam/m/dense_20/bias/Read/ReadVariableOp(Adam/v/dense_20/bias/Read/ReadVariableOp*Adam/m/dense_21/kernel/Read/ReadVariableOp*Adam/v/dense_21/kernel/Read/ReadVariableOp(Adam/m/dense_21/bias/Read/ReadVariableOp(Adam/v/dense_21/bias/Read/ReadVariableOp*Adam/m/dense_22/kernel/Read/ReadVariableOp*Adam/v/dense_22/kernel/Read/ReadVariableOp(Adam/m/dense_22/bias/Read/ReadVariableOp(Adam/v/dense_22/bias/Read/ReadVariableOp*Adam/m/dense_23/kernel/Read/ReadVariableOp*Adam/v/dense_23/kernel/Read/ReadVariableOp(Adam/m/dense_23/bias/Read/ReadVariableOp(Adam/v/dense_23/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst_4*+
Tin$
"2 	*
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
GPU 2J 8� *(
f#R!
__inference__traced_save_258188
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	iterationlearning_rateAdam/m/dense_20/kernelAdam/v/dense_20/kernelAdam/m/dense_20/biasAdam/v/dense_20/biasAdam/m/dense_21/kernelAdam/v/dense_21/kernelAdam/m/dense_21/biasAdam/v/dense_21/biasAdam/m/dense_22/kernelAdam/v/dense_22/kernelAdam/m/dense_22/biasAdam/v/dense_22/biasAdam/m/dense_23/kernelAdam/v/dense_23/kernelAdam/m/dense_23/biasAdam/v/dense_23/biastotal_1count_1totalcount**
Tin#
!2*
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
GPU 2J 8� *+
f&R$
"__inference__traced_restore_258288��	
�
�
(__inference_model_5_layer_call_fn_257728(
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
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_257414o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:m i
'
_output_shapes
:���������
>
_user_specified_name&$inputs_avg_cars_at_home_approx__1_xf:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs_coffee_bar_xf:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_florist_xf:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_low_fat_xf:gc
'
_output_shapes
:���������
8
_user_specified_name inputs_num_children_at_home_xf:`\
'
_output_shapes
:���������
1
_user_specified_nameinputs_prepared_food_xf:\X
'
_output_shapes
:���������
-
_user_specified_nameinputs_salad_bar_xf:kg
'
_output_shapes
:���������
<
_user_specified_name$"inputs_store_sales_in_millions__xf:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs_store_sqft_xf:a	]
'
_output_shapes
:���������
2
_user_specified_nameinputs_total_children_xf:j
f
'
_output_shapes
:���������
;
_user_specified_name#!inputs_unit_sales_in_millions__xf:^Z
'
_output_shapes
:���������
/
_user_specified_nameinputs_video_store_xf
�
�
$__inference_signature_wrapper_257033
examples
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *0
f+R)
'__inference_serve_tf_examples_fn_257002o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:���������
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
�8
�
!__inference__wrapped_model_257079
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
/model_5_dense_20_matmul_readvariableop_resource:	�?
0model_5_dense_20_biasadd_readvariableop_resource:	�B
/model_5_dense_21_matmul_readvariableop_resource:	�>
0model_5_dense_21_biasadd_readvariableop_resource:A
/model_5_dense_22_matmul_readvariableop_resource:>
0model_5_dense_22_biasadd_readvariableop_resource:A
/model_5_dense_23_matmul_readvariableop_resource:>
0model_5_dense_23_biasadd_readvariableop_resource:
identity��'model_5/dense_20/BiasAdd/ReadVariableOp�&model_5/dense_20/MatMul/ReadVariableOp�'model_5/dense_21/BiasAdd/ReadVariableOp�&model_5/dense_21/MatMul/ReadVariableOp�'model_5/dense_22/BiasAdd/ReadVariableOp�&model_5/dense_22/MatMul/ReadVariableOp�'model_5/dense_23/BiasAdd/ReadVariableOp�&model_5/dense_23/MatMul/ReadVariableOpc
!model_5/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_5/concatenate_5/concatConcatV2placeholdercoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfplaceholder_1store_sqft_xftotal_children_xfplaceholder_2video_store_xf*model_5/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
&model_5/dense_20/MatMul/ReadVariableOpReadVariableOp/model_5_dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model_5/dense_20/MatMulMatMul%model_5/concatenate_5/concat:output:0.model_5/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'model_5/dense_20/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_5/dense_20/BiasAddBiasAdd!model_5/dense_20/MatMul:product:0/model_5/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
model_5/dense_20/ReluRelu!model_5/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
&model_5/dense_21/MatMul/ReadVariableOpReadVariableOp/model_5_dense_21_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model_5/dense_21/MatMulMatMul#model_5/dense_20/Relu:activations:0.model_5/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_5/dense_21/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_5/dense_21/BiasAddBiasAdd!model_5/dense_21/MatMul:product:0/model_5/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model_5/dense_21/ReluRelu!model_5/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:����������
&model_5/dense_22/MatMul/ReadVariableOpReadVariableOp/model_5_dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model_5/dense_22/MatMulMatMul#model_5/dense_21/Relu:activations:0.model_5/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_5/dense_22/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_5/dense_22/BiasAddBiasAdd!model_5/dense_22/MatMul:product:0/model_5/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model_5/dense_22/ReluRelu!model_5/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:����������
&model_5/dense_23/MatMul/ReadVariableOpReadVariableOp/model_5_dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model_5/dense_23/MatMulMatMul#model_5/dense_22/Relu:activations:0.model_5/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_5/dense_23/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_5/dense_23/BiasAddBiasAdd!model_5/dense_23/MatMul:product:0/model_5/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
model_5/dense_23/SoftmaxSoftmax!model_5/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"model_5/dense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^model_5/dense_20/BiasAdd/ReadVariableOp'^model_5/dense_20/MatMul/ReadVariableOp(^model_5/dense_21/BiasAdd/ReadVariableOp'^model_5/dense_21/MatMul/ReadVariableOp(^model_5/dense_22/BiasAdd/ReadVariableOp'^model_5/dense_22/MatMul/ReadVariableOp(^model_5/dense_23/BiasAdd/ReadVariableOp'^model_5/dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : 2R
'model_5/dense_20/BiasAdd/ReadVariableOp'model_5/dense_20/BiasAdd/ReadVariableOp2P
&model_5/dense_20/MatMul/ReadVariableOp&model_5/dense_20/MatMul/ReadVariableOp2R
'model_5/dense_21/BiasAdd/ReadVariableOp'model_5/dense_21/BiasAdd/ReadVariableOp2P
&model_5/dense_21/MatMul/ReadVariableOp&model_5/dense_21/MatMul/ReadVariableOp2R
'model_5/dense_22/BiasAdd/ReadVariableOp'model_5/dense_22/BiasAdd/ReadVariableOp2P
&model_5/dense_22/MatMul/ReadVariableOp&model_5/dense_22/MatMul/ReadVariableOp2R
'model_5/dense_23/BiasAdd/ReadVariableOp'model_5/dense_23/BiasAdd/ReadVariableOp2P
&model_5/dense_23/MatMul/ReadVariableOp&model_5/dense_23/MatMul/ReadVariableOp:f b
'
_output_shapes
:���������
7
_user_specified_nameavg_cars_at home(approx).1_xf:VR
'
_output_shapes
:���������
'
_user_specified_namecoffee_bar_xf:SO
'
_output_shapes
:���������
$
_user_specified_name
florist_xf:SO
'
_output_shapes
:���������
$
_user_specified_name
low_fat_xf:`\
'
_output_shapes
:���������
1
_user_specified_namenum_children_at_home_xf:YU
'
_output_shapes
:���������
*
_user_specified_nameprepared_food_xf:UQ
'
_output_shapes
:���������
&
_user_specified_namesalad_bar_xf:d`
'
_output_shapes
:���������
5
_user_specified_namestore_sales(in millions)_xf:VR
'
_output_shapes
:���������
'
_user_specified_namestore_sqft_xf:Z	V
'
_output_shapes
:���������
+
_user_specified_nametotal_children_xf:c
_
'
_output_shapes
:���������
4
_user_specified_nameunit_sales(in millions)_xf:WS
'
_output_shapes
:���������
(
_user_specified_namevideo_store_xf
�

�
D__inference_dense_22_layer_call_and_return_conditional_losses_257943

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�q
�
'__inference_serve_tf_examples_fn_257002
examples%
!transform_features_layer_1_256940%
!transform_features_layer_1_256942%
!transform_features_layer_1_256944%
!transform_features_layer_1_256946B
/model_5_dense_20_matmul_readvariableop_resource:	�?
0model_5_dense_20_biasadd_readvariableop_resource:	�B
/model_5_dense_21_matmul_readvariableop_resource:	�>
0model_5_dense_21_biasadd_readvariableop_resource:A
/model_5_dense_22_matmul_readvariableop_resource:>
0model_5_dense_22_biasadd_readvariableop_resource:A
/model_5_dense_23_matmul_readvariableop_resource:>
0model_5_dense_23_biasadd_readvariableop_resource:
identity��'model_5/dense_20/BiasAdd/ReadVariableOp�&model_5/dense_20/MatMul/ReadVariableOp�'model_5/dense_21/BiasAdd/ReadVariableOp�&model_5/dense_21/MatMul/ReadVariableOp�'model_5/dense_22/BiasAdd/ReadVariableOp�&model_5/dense_22/MatMul/ReadVariableOp�'model_5/dense_23/BiasAdd/ReadVariableOp�&model_5/dense_23/MatMul/ReadVariableOp�2transform_features_layer_1/StatefulPartitionedCallU
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
valueB �
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*�
value�B�Bavg_cars_at home(approx).1B
coffee_barBfloristBlow_fatBnum_children_at_homeBprepared_foodB	salad_barBstore_sales(in millions)B
store_sqftBtotal_childrenBunit_sales(in millions)Bvideo_storej
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB �
ParseExample/ParseExampleV2ParseExampleV2examples*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0ParseExample/Const_1:output:0ParseExample/Const_2:output:0ParseExample/Const_3:output:0ParseExample/Const_4:output:0ParseExample/Const_5:output:0ParseExample/Const_6:output:0ParseExample/Const_7:output:0ParseExample/Const_8:output:0ParseExample/Const_9:output:0ParseExample/Const_10:output:0ParseExample/Const_11:output:0*
Tdense
2*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*Z
dense_shapesJ
H::::::::::::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 z
 transform_features_layer_1/ShapeShape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:x
.transform_features_layer_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0transform_features_layer_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0transform_features_layer_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(transform_features_layer_1/strided_sliceStridedSlice)transform_features_layer_1/Shape:output:07transform_features_layer_1/strided_slice/stack:output:09transform_features_layer_1/strided_slice/stack_1:output:09transform_features_layer_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
"transform_features_layer_1/Shape_1Shape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:z
0transform_features_layer_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2transform_features_layer_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2transform_features_layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*transform_features_layer_1/strided_slice_1StridedSlice+transform_features_layer_1/Shape_1:output:09transform_features_layer_1/strided_slice_1/stack:output:0;transform_features_layer_1/strided_slice_1/stack_1:output:0;transform_features_layer_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)transform_features_layer_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
'transform_features_layer_1/zeros/packedPack3transform_features_layer_1/strided_slice_1:output:02transform_features_layer_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:h
&transform_features_layer_1/zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R �
 transform_features_layer_1/zerosFill0transform_features_layer_1/zeros/packed:output:0/transform_features_layer_1/zeros/Const:output:0*
T0	*'
_output_shapes
:����������
1transform_features_layer_1/PlaceholderWithDefaultPlaceholderWithDefault)transform_features_layer_1/zeros:output:0*'
_output_shapes
:���������*
dtype0	*
shape:����������	
2transform_features_layer_1/StatefulPartitionedCallStatefulPartitionedCall*ParseExample/ParseExampleV2:dense_values:0*ParseExample/ParseExampleV2:dense_values:1:transform_features_layer_1/PlaceholderWithDefault:output:0*ParseExample/ParseExampleV2:dense_values:2*ParseExample/ParseExampleV2:dense_values:3*ParseExample/ParseExampleV2:dense_values:4*ParseExample/ParseExampleV2:dense_values:5*ParseExample/ParseExampleV2:dense_values:6*ParseExample/ParseExampleV2:dense_values:7*ParseExample/ParseExampleV2:dense_values:8*ParseExample/ParseExampleV2:dense_values:9+ParseExample/ParseExampleV2:dense_values:10+ParseExample/ParseExampleV2:dense_values:11!transform_features_layer_1_256940!transform_features_layer_1_256942!transform_features_layer_1_256944!transform_features_layer_1_256946*
Tin
2	*
Tout
2											*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *"
fR
__inference_pruned_256833�
model_5/CastCast;transform_features_layer_1/StatefulPartitionedCall:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_5/Cast_1Cast;transform_features_layer_1/StatefulPartitionedCall:output:1*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_5/Cast_2Cast;transform_features_layer_1/StatefulPartitionedCall:output:3*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_5/Cast_3Cast;transform_features_layer_1/StatefulPartitionedCall:output:4*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_5/Cast_4Cast;transform_features_layer_1/StatefulPartitionedCall:output:5*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_5/Cast_5Cast;transform_features_layer_1/StatefulPartitionedCall:output:6*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_5/Cast_6Cast;transform_features_layer_1/StatefulPartitionedCall:output:7*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_5/Cast_7Cast<transform_features_layer_1/StatefulPartitionedCall:output:10*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_5/Cast_8Cast<transform_features_layer_1/StatefulPartitionedCall:output:11*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_5/Cast_9Cast<transform_features_layer_1/StatefulPartitionedCall:output:12*

DstT0*

SrcT0	*'
_output_shapes
:���������c
!model_5/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_5/concatenate_5/concatConcatV2model_5/Cast:y:0model_5/Cast_1:y:0model_5/Cast_2:y:0model_5/Cast_3:y:0model_5/Cast_4:y:0model_5/Cast_5:y:0model_5/Cast_6:y:0;transform_features_layer_1/StatefulPartitionedCall:output:8;transform_features_layer_1/StatefulPartitionedCall:output:9model_5/Cast_7:y:0model_5/Cast_8:y:0model_5/Cast_9:y:0*model_5/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
&model_5/dense_20/MatMul/ReadVariableOpReadVariableOp/model_5_dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model_5/dense_20/MatMulMatMul%model_5/concatenate_5/concat:output:0.model_5/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'model_5/dense_20/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_5/dense_20/BiasAddBiasAdd!model_5/dense_20/MatMul:product:0/model_5/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
model_5/dense_20/ReluRelu!model_5/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
&model_5/dense_21/MatMul/ReadVariableOpReadVariableOp/model_5_dense_21_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model_5/dense_21/MatMulMatMul#model_5/dense_20/Relu:activations:0.model_5/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_5/dense_21/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_5/dense_21/BiasAddBiasAdd!model_5/dense_21/MatMul:product:0/model_5/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model_5/dense_21/ReluRelu!model_5/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:����������
&model_5/dense_22/MatMul/ReadVariableOpReadVariableOp/model_5_dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model_5/dense_22/MatMulMatMul#model_5/dense_21/Relu:activations:0.model_5/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_5/dense_22/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_5/dense_22/BiasAddBiasAdd!model_5/dense_22/MatMul:product:0/model_5/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model_5/dense_22/ReluRelu!model_5/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:����������
&model_5/dense_23/MatMul/ReadVariableOpReadVariableOp/model_5_dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model_5/dense_23/MatMulMatMul#model_5/dense_22/Relu:activations:0.model_5/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_5/dense_23/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_5/dense_23/BiasAddBiasAdd!model_5/dense_23/MatMul:product:0/model_5/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
model_5/dense_23/SoftmaxSoftmax!model_5/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"model_5/dense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^model_5/dense_20/BiasAdd/ReadVariableOp'^model_5/dense_20/MatMul/ReadVariableOp(^model_5/dense_21/BiasAdd/ReadVariableOp'^model_5/dense_21/MatMul/ReadVariableOp(^model_5/dense_22/BiasAdd/ReadVariableOp'^model_5/dense_22/MatMul/ReadVariableOp(^model_5/dense_23/BiasAdd/ReadVariableOp'^model_5/dense_23/MatMul/ReadVariableOp3^transform_features_layer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : : : 2R
'model_5/dense_20/BiasAdd/ReadVariableOp'model_5/dense_20/BiasAdd/ReadVariableOp2P
&model_5/dense_20/MatMul/ReadVariableOp&model_5/dense_20/MatMul/ReadVariableOp2R
'model_5/dense_21/BiasAdd/ReadVariableOp'model_5/dense_21/BiasAdd/ReadVariableOp2P
&model_5/dense_21/MatMul/ReadVariableOp&model_5/dense_21/MatMul/ReadVariableOp2R
'model_5/dense_22/BiasAdd/ReadVariableOp'model_5/dense_22/BiasAdd/ReadVariableOp2P
&model_5/dense_22/MatMul/ReadVariableOp&model_5/dense_22/MatMul/ReadVariableOp2R
'model_5/dense_23/BiasAdd/ReadVariableOp'model_5/dense_23/BiasAdd/ReadVariableOp2P
&model_5/dense_23/MatMul/ReadVariableOp&model_5/dense_23/MatMul/ReadVariableOp2h
2transform_features_layer_1/StatefulPartitionedCall2transform_features_layer_1/StatefulPartitionedCall:M I
#
_output_shapes
:���������
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
�

�
D__inference_dense_22_layer_call_and_return_conditional_losses_257390

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_dense_21_layer_call_fn_257912

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_257373o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�6
�	
C__inference_model_5_layer_call_and_return_conditional_losses_257805(
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
'dense_20_matmul_readvariableop_resource:	�7
(dense_20_biasadd_readvariableop_resource:	�:
'dense_21_matmul_readvariableop_resource:	�6
(dense_21_biasadd_readvariableop_resource:9
'dense_22_matmul_readvariableop_resource:6
(dense_22_biasadd_readvariableop_resource:9
'dense_23_matmul_readvariableop_resource:6
(dense_23_biasadd_readvariableop_resource:
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_5/concatConcatV2$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xf"concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_20/MatMulMatMulconcatenate_5/concat:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_21/MatMulMatMuldense_20/Relu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:m i
'
_output_shapes
:���������
>
_user_specified_name&$inputs_avg_cars_at_home_approx__1_xf:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs_coffee_bar_xf:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_florist_xf:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_low_fat_xf:gc
'
_output_shapes
:���������
8
_user_specified_name inputs_num_children_at_home_xf:`\
'
_output_shapes
:���������
1
_user_specified_nameinputs_prepared_food_xf:\X
'
_output_shapes
:���������
-
_user_specified_nameinputs_salad_bar_xf:kg
'
_output_shapes
:���������
<
_user_specified_name$"inputs_store_sales_in_millions__xf:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs_store_sqft_xf:a	]
'
_output_shapes
:���������
2
_user_specified_nameinputs_total_children_xf:j
f
'
_output_shapes
:���������
;
_user_specified_name#!inputs_unit_sales_in_millions__xf:^Z
'
_output_shapes
:���������
/
_user_specified_nameinputs_video_store_xf
�&
�
C__inference_model_5_layer_call_and_return_conditional_losses_257694
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
dense_20_257673:	�
dense_20_257675:	�"
dense_21_257678:	�
dense_21_257680:!
dense_22_257683:
dense_22_257685:!
dense_23_257688:
dense_23_257690:
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�
concatenate_5/PartitionedCallPartitionedCallplaceholdercoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfplaceholder_1store_sqft_xftotal_children_xfplaceholder_2video_store_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_257343�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_20_257673dense_20_257675*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_257356�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_257678dense_21_257680*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_257373�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_257683dense_22_257685*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_257390�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_257688dense_23_257690*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_257407x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:f b
'
_output_shapes
:���������
7
_user_specified_nameavg_cars_at home(approx).1_xf:VR
'
_output_shapes
:���������
'
_user_specified_namecoffee_bar_xf:SO
'
_output_shapes
:���������
$
_user_specified_name
florist_xf:SO
'
_output_shapes
:���������
$
_user_specified_name
low_fat_xf:`\
'
_output_shapes
:���������
1
_user_specified_namenum_children_at_home_xf:YU
'
_output_shapes
:���������
*
_user_specified_nameprepared_food_xf:UQ
'
_output_shapes
:���������
&
_user_specified_namesalad_bar_xf:d`
'
_output_shapes
:���������
5
_user_specified_namestore_sales(in millions)_xf:VR
'
_output_shapes
:���������
'
_user_specified_namestore_sqft_xf:Z	V
'
_output_shapes
:���������
+
_user_specified_nametotal_children_xf:c
_
'
_output_shapes
:���������
4
_user_specified_nameunit_sales(in millions)_xf:WS
'
_output_shapes
:���������
(
_user_specified_namevideo_store_xf
�
�
(__inference_model_5_layer_call_fn_257622
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
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallplaceholdercoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfplaceholder_1store_sqft_xftotal_children_xfplaceholder_2video_store_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_257571o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
'
_output_shapes
:���������
7
_user_specified_nameavg_cars_at home(approx).1_xf:VR
'
_output_shapes
:���������
'
_user_specified_namecoffee_bar_xf:SO
'
_output_shapes
:���������
$
_user_specified_name
florist_xf:SO
'
_output_shapes
:���������
$
_user_specified_name
low_fat_xf:`\
'
_output_shapes
:���������
1
_user_specified_namenum_children_at_home_xf:YU
'
_output_shapes
:���������
*
_user_specified_nameprepared_food_xf:UQ
'
_output_shapes
:���������
&
_user_specified_namesalad_bar_xf:d`
'
_output_shapes
:���������
5
_user_specified_namestore_sales(in millions)_xf:VR
'
_output_shapes
:���������
'
_user_specified_namestore_sqft_xf:Z	V
'
_output_shapes
:���������
+
_user_specified_nametotal_children_xf:c
_
'
_output_shapes
:���������
4
_user_specified_nameunit_sales(in millions)_xf:WS
'
_output_shapes
:���������
(
_user_specified_namevideo_store_xf
�

�
D__inference_dense_23_layer_call_and_return_conditional_losses_257407

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_dense_21_layer_call_and_return_conditional_losses_257923

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_dense_20_layer_call_and_return_conditional_losses_257356

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
;__inference_transform_features_layer_1_layer_call_fn_257189
placeholder

coffee_bar
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
identity_11	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallplaceholder
coffee_barfloristlow_fatnum_children_at_homeprepared_food	salad_barplaceholder_1
store_sqfttotal_childrenplaceholder_2video_storeunknown	unknown_0	unknown_1	unknown_2*
Tin
2*
Tout
2										*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *_
fZRX
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_257156o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0	*'
_output_shapes
:���������q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0	*'
_output_shapes
:���������q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:���������q

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0	*'
_output_shapes
:���������q

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0	*'
_output_shapes
:���������q

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:���������q

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:���������q

Identity_9Identity StatefulPartitionedCall:output:9^NoOp*
T0	*'
_output_shapes
:���������s
Identity_10Identity!StatefulPartitionedCall:output:10^NoOp*
T0	*'
_output_shapes
:���������s
Identity_11Identity!StatefulPartitionedCall:output:11^NoOp*
T0	*'
_output_shapes
:���������`
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
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
'
_output_shapes
:���������
4
_user_specified_nameavg_cars_at home(approx).1:SO
'
_output_shapes
:���������
$
_user_specified_name
coffee_bar:PL
'
_output_shapes
:���������
!
_user_specified_name	florist:PL
'
_output_shapes
:���������
!
_user_specified_name	low_fat:]Y
'
_output_shapes
:���������
.
_user_specified_namenum_children_at_home:VR
'
_output_shapes
:���������
'
_user_specified_nameprepared_food:RN
'
_output_shapes
:���������
#
_user_specified_name	salad_bar:a]
'
_output_shapes
:���������
2
_user_specified_namestore_sales(in millions):SO
'
_output_shapes
:���������
$
_user_specified_name
store_sqft:W	S
'
_output_shapes
:���������
(
_user_specified_nametotal_children:`
\
'
_output_shapes
:���������
1
_user_specified_nameunit_sales(in millions):TP
'
_output_shapes
:���������
%
_user_specified_namevideo_store:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
)__inference_dense_22_layer_call_fn_257932

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_257390o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�6
�	
C__inference_model_5_layer_call_and_return_conditional_losses_257850(
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
'dense_20_matmul_readvariableop_resource:	�7
(dense_20_biasadd_readvariableop_resource:	�:
'dense_21_matmul_readvariableop_resource:	�6
(dense_21_biasadd_readvariableop_resource:9
'dense_22_matmul_readvariableop_resource:6
(dense_22_biasadd_readvariableop_resource:9
'dense_23_matmul_readvariableop_resource:6
(dense_23_biasadd_readvariableop_resource:
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_5/concatConcatV2$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xf"concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_20/MatMulMatMulconcatenate_5/concat:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_21/MatMulMatMuldense_20/Relu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:m i
'
_output_shapes
:���������
>
_user_specified_name&$inputs_avg_cars_at_home_approx__1_xf:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs_coffee_bar_xf:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_florist_xf:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_low_fat_xf:gc
'
_output_shapes
:���������
8
_user_specified_name inputs_num_children_at_home_xf:`\
'
_output_shapes
:���������
1
_user_specified_nameinputs_prepared_food_xf:\X
'
_output_shapes
:���������
-
_user_specified_nameinputs_salad_bar_xf:kg
'
_output_shapes
:���������
<
_user_specified_name$"inputs_store_sales_in_millions__xf:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs_store_sqft_xf:a	]
'
_output_shapes
:���������
2
_user_specified_nameinputs_total_children_xf:j
f
'
_output_shapes
:���������
;
_user_specified_name#!inputs_unit_sales_in_millions__xf:^Z
'
_output_shapes
:���������
/
_user_specified_nameinputs_video_store_xf
�5
�
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_258071%
!inputs_avg_cars_at_home_approx__1
inputs_coffee_bar
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
identity_11	��StatefulPartitionedCallV
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
valueB:�
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
valueB:�
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
:����������
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:���������*
dtype0	*
shape:����������
StatefulPartitionedCallStatefulPartitionedCall!inputs_avg_cars_at_home_approx__1inputs_coffee_barPlaceholderWithDefault:output:0inputs_floristinputs_low_fatinputs_num_children_at_homeinputs_prepared_foodinputs_salad_barinputs_store_sales_in_millions_inputs_store_sqftinputs_total_childreninputs_unit_sales_in_millions_inputs_video_storeunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2											*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *"
fR
__inference_pruned_256833o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:3^NoOp*
T0	*'
_output_shapes
:���������q

Identity_3Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:���������q

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0	*'
_output_shapes
:���������q

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0	*'
_output_shapes
:���������q

Identity_6Identity StatefulPartitionedCall:output:7^NoOp*
T0	*'
_output_shapes
:���������q

Identity_7Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:���������q

Identity_8Identity StatefulPartitionedCall:output:9^NoOp*
T0*'
_output_shapes
:���������r

Identity_9Identity!StatefulPartitionedCall:output:10^NoOp*
T0	*'
_output_shapes
:���������s
Identity_10Identity!StatefulPartitionedCall:output:11^NoOp*
T0	*'
_output_shapes
:���������s
Identity_11Identity!StatefulPartitionedCall:output:12^NoOp*
T0	*'
_output_shapes
:���������`
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
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
'
_output_shapes
:���������
;
_user_specified_name#!inputs_avg_cars_at_home_approx__1:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_coffee_bar:WS
'
_output_shapes
:���������
(
_user_specified_nameinputs_florist:WS
'
_output_shapes
:���������
(
_user_specified_nameinputs_low_fat:d`
'
_output_shapes
:���������
5
_user_specified_nameinputs_num_children_at_home:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs_prepared_food:YU
'
_output_shapes
:���������
*
_user_specified_nameinputs_salad_bar:hd
'
_output_shapes
:���������
9
_user_specified_name!inputs_store_sales_in_millions_:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_store_sqft:^	Z
'
_output_shapes
:���������
/
_user_specified_nameinputs_total_children:g
c
'
_output_shapes
:���������
8
_user_specified_name inputs_unit_sales_in_millions_:[W
'
_output_shapes
:���������
,
_user_specified_nameinputs_video_store:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
.__inference_concatenate_5_layer_call_fn_257866
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
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_257343`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_8:Q	M
'
_output_shapes
:���������
"
_user_specified_name
inputs_9:R
N
'
_output_shapes
:���������
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_11
�
�
)__inference_dense_23_layer_call_fn_257952

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_257407o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_dense_23_layer_call_and_return_conditional_losses_257963

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
C__inference_model_5_layer_call_and_return_conditional_losses_257658
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
dense_20_257637:	�
dense_20_257639:	�"
dense_21_257642:	�
dense_21_257644:!
dense_22_257647:
dense_22_257649:!
dense_23_257652:
dense_23_257654:
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�
concatenate_5/PartitionedCallPartitionedCallplaceholdercoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfplaceholder_1store_sqft_xftotal_children_xfplaceholder_2video_store_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_257343�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_20_257637dense_20_257639*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_257356�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_257642dense_21_257644*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_257373�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_257647dense_22_257649*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_257390�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_257652dense_23_257654*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_257407x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:f b
'
_output_shapes
:���������
7
_user_specified_nameavg_cars_at home(approx).1_xf:VR
'
_output_shapes
:���������
'
_user_specified_namecoffee_bar_xf:SO
'
_output_shapes
:���������
$
_user_specified_name
florist_xf:SO
'
_output_shapes
:���������
$
_user_specified_name
low_fat_xf:`\
'
_output_shapes
:���������
1
_user_specified_namenum_children_at_home_xf:YU
'
_output_shapes
:���������
*
_user_specified_nameprepared_food_xf:UQ
'
_output_shapes
:���������
&
_user_specified_namesalad_bar_xf:d`
'
_output_shapes
:���������
5
_user_specified_namestore_sales(in millions)_xf:VR
'
_output_shapes
:���������
'
_user_specified_namestore_sqft_xf:Z	V
'
_output_shapes
:���������
+
_user_specified_nametotal_children_xf:c
_
'
_output_shapes
:���������
4
_user_specified_nameunit_sales(in millions)_xf:WS
'
_output_shapes
:���������
(
_user_specified_namevideo_store_xf
�2
�
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_257297
placeholder

coffee_bar
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
identity_11	��StatefulPartitionedCall@
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
valueB:�
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
valueB:�
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
:����������
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:���������*
dtype0	*
shape:����������
StatefulPartitionedCallStatefulPartitionedCallplaceholder
coffee_barPlaceholderWithDefault:output:0floristlow_fatnum_children_at_homeprepared_food	salad_barplaceholder_1
store_sqfttotal_childrenplaceholder_2video_storeunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2											*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *"
fR
__inference_pruned_256833o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:3^NoOp*
T0	*'
_output_shapes
:���������q

Identity_3Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:���������q

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0	*'
_output_shapes
:���������q

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0	*'
_output_shapes
:���������q

Identity_6Identity StatefulPartitionedCall:output:7^NoOp*
T0	*'
_output_shapes
:���������q

Identity_7Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:���������q

Identity_8Identity StatefulPartitionedCall:output:9^NoOp*
T0*'
_output_shapes
:���������r

Identity_9Identity!StatefulPartitionedCall:output:10^NoOp*
T0	*'
_output_shapes
:���������s
Identity_10Identity!StatefulPartitionedCall:output:11^NoOp*
T0	*'
_output_shapes
:���������s
Identity_11Identity!StatefulPartitionedCall:output:12^NoOp*
T0	*'
_output_shapes
:���������`
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
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
'
_output_shapes
:���������
4
_user_specified_nameavg_cars_at home(approx).1:SO
'
_output_shapes
:���������
$
_user_specified_name
coffee_bar:PL
'
_output_shapes
:���������
!
_user_specified_name	florist:PL
'
_output_shapes
:���������
!
_user_specified_name	low_fat:]Y
'
_output_shapes
:���������
.
_user_specified_namenum_children_at_home:VR
'
_output_shapes
:���������
'
_user_specified_nameprepared_food:RN
'
_output_shapes
:���������
#
_user_specified_name	salad_bar:a]
'
_output_shapes
:���������
2
_user_specified_namestore_sales(in millions):SO
'
_output_shapes
:���������
$
_user_specified_name
store_sqft:W	S
'
_output_shapes
:���������
(
_user_specified_nametotal_children:`
\
'
_output_shapes
:���������
1
_user_specified_nameunit_sales(in millions):TP
'
_output_shapes
:���������
%
_user_specified_namevideo_store:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
)__inference_dense_20_layer_call_fn_257892

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_257356p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�
__inference__traced_save_258188
file_prefix.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop5
1savev2_adam_m_dense_20_kernel_read_readvariableop5
1savev2_adam_v_dense_20_kernel_read_readvariableop3
/savev2_adam_m_dense_20_bias_read_readvariableop3
/savev2_adam_v_dense_20_bias_read_readvariableop5
1savev2_adam_m_dense_21_kernel_read_readvariableop5
1savev2_adam_v_dense_21_kernel_read_readvariableop3
/savev2_adam_m_dense_21_bias_read_readvariableop3
/savev2_adam_v_dense_21_bias_read_readvariableop5
1savev2_adam_m_dense_22_kernel_read_readvariableop5
1savev2_adam_v_dense_22_kernel_read_readvariableop3
/savev2_adam_m_dense_22_bias_read_readvariableop3
/savev2_adam_v_dense_22_bias_read_readvariableop5
1savev2_adam_m_dense_23_kernel_read_readvariableop5
1savev2_adam_v_dense_23_kernel_read_readvariableop3
/savev2_adam_m_dense_23_bias_read_readvariableop3
/savev2_adam_v_dense_23_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const_4

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop1savev2_adam_m_dense_20_kernel_read_readvariableop1savev2_adam_v_dense_20_kernel_read_readvariableop/savev2_adam_m_dense_20_bias_read_readvariableop/savev2_adam_v_dense_20_bias_read_readvariableop1savev2_adam_m_dense_21_kernel_read_readvariableop1savev2_adam_v_dense_21_kernel_read_readvariableop/savev2_adam_m_dense_21_bias_read_readvariableop/savev2_adam_v_dense_21_bias_read_readvariableop1savev2_adam_m_dense_22_kernel_read_readvariableop1savev2_adam_v_dense_22_kernel_read_readvariableop/savev2_adam_m_dense_22_bias_read_readvariableop/savev2_adam_v_dense_22_bias_read_readvariableop1savev2_adam_m_dense_23_kernel_read_readvariableop1savev2_adam_v_dense_23_kernel_read_readvariableop/savev2_adam_m_dense_23_bias_read_readvariableop/savev2_adam_v_dense_23_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const_4"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *-
dtypes#
!2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�:�:	�:::::: : :	�:	�:�:�:	�:	�::::::::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :%!

_output_shapes
:	�:%!

_output_shapes
:	�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�:%!

_output_shapes
:	�: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
C__inference_model_5_layer_call_and_return_conditional_losses_257571

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
dense_20_257550:	�
dense_20_257552:	�"
dense_21_257555:	�
dense_21_257557:!
dense_22_257560:
dense_22_257562:!
dense_23_257565:
dense_23_257567:
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�
concatenate_5/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_257343�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_20_257550dense_20_257552*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_257356�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_257555dense_21_257557*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_257373�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_257560dense_22_257562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_257390�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_257565dense_23_257567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_257407x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:O	K
'
_output_shapes
:���������
 
_user_specified_nameinputs:O
K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
"__inference__traced_restore_258288
file_prefix3
 assignvariableop_dense_20_kernel:	�/
 assignvariableop_1_dense_20_bias:	�5
"assignvariableop_2_dense_21_kernel:	�.
 assignvariableop_3_dense_21_bias:4
"assignvariableop_4_dense_22_kernel:.
 assignvariableop_5_dense_22_bias:4
"assignvariableop_6_dense_23_kernel:.
 assignvariableop_7_dense_23_bias:&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: =
*assignvariableop_10_adam_m_dense_20_kernel:	�=
*assignvariableop_11_adam_v_dense_20_kernel:	�7
(assignvariableop_12_adam_m_dense_20_bias:	�7
(assignvariableop_13_adam_v_dense_20_bias:	�=
*assignvariableop_14_adam_m_dense_21_kernel:	�=
*assignvariableop_15_adam_v_dense_21_kernel:	�6
(assignvariableop_16_adam_m_dense_21_bias:6
(assignvariableop_17_adam_v_dense_21_bias:<
*assignvariableop_18_adam_m_dense_22_kernel:<
*assignvariableop_19_adam_v_dense_22_kernel:6
(assignvariableop_20_adam_m_dense_22_bias:6
(assignvariableop_21_adam_v_dense_22_bias:<
*assignvariableop_22_adam_m_dense_23_kernel:<
*assignvariableop_23_adam_v_dense_23_kernel:6
(assignvariableop_24_adam_m_dense_23_bias:6
(assignvariableop_25_adam_v_dense_23_bias:%
assignvariableop_26_total_1: %
assignvariableop_27_count_1: #
assignvariableop_28_total: #
assignvariableop_29_count: 
identity_31��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_20_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_20_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_21_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_21_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_22_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_22_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_23_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_23_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp*assignvariableop_10_adam_m_dense_20_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp*assignvariableop_11_adam_v_dense_20_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp(assignvariableop_12_adam_m_dense_20_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_v_dense_20_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_m_dense_21_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_v_dense_21_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_m_dense_21_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_v_dense_21_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_m_dense_22_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_v_dense_22_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_m_dense_22_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_v_dense_22_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_m_dense_23_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_v_dense_23_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_m_dense_23_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_v_dense_23_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
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
�

�
D__inference_dense_21_layer_call_and_return_conditional_losses_257373

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�k
�
__inference_pruned_256833

inputs
inputs_1
inputs_2	
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12-
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
identity_12	�c
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
valueB: �
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*:
value1B/ B)x (scale_to_0_1_1/min_and_max/Shape:0) = �
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
valueB: �
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*8
value/B- B'x (scale_to_0_1/min_and_max/Shape:0) = �
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
 *  �?Y
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
 *  �?[
scale_to_0_1_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Q
inputs_copyIdentityinputs*
T0*'
_output_shapes
:���������c
CastCastinputs_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:����������
/scale_to_0_1_1/min_and_max/assert_equal_1/EqualEqual)scale_to_0_1_1/min_and_max/Shape:output:0+scale_to_0_1_1/min_and_max/Shape_1:output:0*
T0*
_output_shapes
: �
-scale_to_0_1_1/min_and_max/assert_equal_1/AllAll3scale_to_0_1_1/min_and_max/assert_equal_1/Equal:z:08scale_to_0_1_1/min_and_max/assert_equal_1/Const:output:0*
_output_shapes
: �
-scale_to_0_1/min_and_max/assert_equal_1/EqualEqual'scale_to_0_1/min_and_max/Shape:output:0)scale_to_0_1/min_and_max/Shape_1:output:0*
T0*
_output_shapes
: �
+scale_to_0_1/min_and_max/assert_equal_1/AllAll1scale_to_0_1/min_and_max/assert_equal_1/Equal:z:06scale_to_0_1/min_and_max/assert_equal_1/Const:output:0*
_output_shapes
: �
5scale_to_0_1/min_and_max/assert_equal_1/Assert/AssertAssert4scale_to_0_1/min_and_max/assert_equal_1/All:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_0:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_1:output:0'scale_to_0_1/min_and_max/Shape:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_3:output:0)scale_to_0_1/min_and_max/Shape_1:output:0*
T	
2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
7scale_to_0_1_1/min_and_max/assert_equal_1/Assert/AssertAssert6scale_to_0_1_1/min_and_max/assert_equal_1/All:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_0:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_1:output:0)scale_to_0_1_1/min_and_max/Shape:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_3:output:0+scale_to_0_1_1/min_and_max/Shape_1:output:06^scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert*
T	
2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
NoOpNoOp6^scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert8^scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert*"
_acd_function_control_output(*&
 _has_manual_control_dependencies(*
_output_shapes
 W
IdentityIdentityCast:y:0^NoOp*
T0	*'
_output_shapes
:���������U
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:���������g
Cast_1Castinputs_1_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:���������[

Identity_1Identity
Cast_1:y:0^NoOp*
T0	*'
_output_shapes
:���������U
inputs_2_copyIdentityinputs_2*
T0	*'
_output_shapes
:���������g

Identity_2Identityinputs_2_copy:output:0^NoOp*
T0	*'
_output_shapes
:���������U
inputs_3_copyIdentityinputs_3*
T0*'
_output_shapes
:���������g
Cast_2Castinputs_3_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:���������[

Identity_3Identity
Cast_2:y:0^NoOp*
T0	*'
_output_shapes
:���������U
inputs_4_copyIdentityinputs_4*
T0*'
_output_shapes
:���������g
Cast_8Castinputs_4_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:���������[

Identity_4Identity
Cast_8:y:0^NoOp*
T0	*'
_output_shapes
:���������U
inputs_5_copyIdentityinputs_5*
T0*'
_output_shapes
:���������g
Cast_7Castinputs_5_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:���������[

Identity_5Identity
Cast_7:y:0^NoOp*
T0	*'
_output_shapes
:���������U
inputs_6_copyIdentityinputs_6*
T0*'
_output_shapes
:���������g
Cast_9Castinputs_6_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:���������[

Identity_6Identity
Cast_9:y:0^NoOp*
T0	*'
_output_shapes
:���������U
inputs_7_copyIdentityinputs_7*
T0*'
_output_shapes
:���������g
Cast_3Castinputs_7_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:���������[

Identity_7Identity
Cast_3:y:0^NoOp*
T0	*'
_output_shapes
:���������U
inputs_8_copyIdentityinputs_8*
T0*'
_output_shapes
:���������{
#scale_to_0_1/min_and_max/Identity_2Identity)scale_to_0_1_min_and_max_identity_2_input*
T0*
_output_shapes
: �
scale_to_0_1/min_and_max/sub_1Sub)scale_to_0_1/min_and_max/sub_1/x:output:0,scale_to_0_1/min_and_max/Identity_2:output:0*
T0*
_output_shapes
: �
scale_to_0_1/subSubinputs_8_copy:output:0"scale_to_0_1/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������l
scale_to_0_1/zeros_like	ZerosLikescale_to_0_1/sub:z:0*
T0*'
_output_shapes
:���������{
#scale_to_0_1/min_and_max/Identity_3Identity)scale_to_0_1_min_and_max_identity_3_input*
T0*
_output_shapes
: �
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
:���������r
scale_to_0_1/Cast_1Castscale_to_0_1/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:����������
scale_to_0_1/sub_1Sub,scale_to_0_1/min_and_max/Identity_3:output:0"scale_to_0_1/min_and_max/sub_1:z:0*
T0*
_output_shapes
: 
scale_to_0_1/truedivRealDivscale_to_0_1/sub:z:0scale_to_0_1/sub_1:z:0*
T0*'
_output_shapes
:���������i
scale_to_0_1/SigmoidSigmoidinputs_8_copy:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1/SelectV2SelectV2scale_to_0_1/Cast_1:y:0scale_to_0_1/truediv:z:0scale_to_0_1/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1/mulMulscale_to_0_1/SelectV2:output:0scale_to_0_1/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1/add_1AddV2scale_to_0_1/mul:z:0scale_to_0_1/add_1/y:output:0*
T0*'
_output_shapes
:���������g

Identity_8Identityscale_to_0_1/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������U
inputs_9_copyIdentityinputs_9*
T0*'
_output_shapes
:���������
%scale_to_0_1_1/min_and_max/Identity_2Identity+scale_to_0_1_1_min_and_max_identity_2_input*
T0*
_output_shapes
: �
 scale_to_0_1_1/min_and_max/sub_1Sub+scale_to_0_1_1/min_and_max/sub_1/x:output:0.scale_to_0_1_1/min_and_max/Identity_2:output:0*
T0*
_output_shapes
: �
scale_to_0_1_1/subSubinputs_9_copy:output:0$scale_to_0_1_1/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������p
scale_to_0_1_1/zeros_like	ZerosLikescale_to_0_1_1/sub:z:0*
T0*'
_output_shapes
:���������
%scale_to_0_1_1/min_and_max/Identity_3Identity+scale_to_0_1_1_min_and_max_identity_3_input*
T0*
_output_shapes
: �
scale_to_0_1_1/LessLess$scale_to_0_1_1/min_and_max/sub_1:z:0.scale_to_0_1_1/min_and_max/Identity_3:output:0*
T0*
_output_shapes
: d
scale_to_0_1_1/CastCastscale_to_0_1_1/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1_1/addAddV2scale_to_0_1_1/zeros_like:y:0scale_to_0_1_1/Cast:y:0*
T0*'
_output_shapes
:���������v
scale_to_0_1_1/Cast_1Castscale_to_0_1_1/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:����������
scale_to_0_1_1/sub_1Sub.scale_to_0_1_1/min_and_max/Identity_3:output:0$scale_to_0_1_1/min_and_max/sub_1:z:0*
T0*
_output_shapes
: �
scale_to_0_1_1/truedivRealDivscale_to_0_1_1/sub:z:0scale_to_0_1_1/sub_1:z:0*
T0*'
_output_shapes
:���������k
scale_to_0_1_1/SigmoidSigmoidinputs_9_copy:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_1/SelectV2SelectV2scale_to_0_1_1/Cast_1:y:0scale_to_0_1_1/truediv:z:0scale_to_0_1_1/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_1/mulMul scale_to_0_1_1/SelectV2:output:0scale_to_0_1_1/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_1/add_1AddV2scale_to_0_1_1/mul:z:0scale_to_0_1_1/add_1/y:output:0*
T0*'
_output_shapes
:���������i

Identity_9Identityscale_to_0_1_1/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������W
inputs_10_copyIdentity	inputs_10*
T0*'
_output_shapes
:���������h
Cast_6Castinputs_10_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:���������\
Identity_10Identity
Cast_6:y:0^NoOp*
T0	*'
_output_shapes
:���������W
inputs_11_copyIdentity	inputs_11*
T0*'
_output_shapes
:���������h
Cast_4Castinputs_11_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:���������\
Identity_11Identity
Cast_4:y:0^NoOp*
T0	*'
_output_shapes
:���������W
inputs_12_copyIdentity	inputs_12*
T0*'
_output_shapes
:���������h
Cast_5Castinputs_12_copy:output:0*

DstT0	*

SrcT0*'
_output_shapes
:���������\
Identity_12Identity
Cast_5:y:0^NoOp*
T0	*'
_output_shapes
:���������"
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
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : :- )
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-	)
'
_output_shapes
:���������:-
)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:
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
�'
�
;__inference_transform_features_layer_1_layer_call_fn_258009%
!inputs_avg_cars_at_home_approx__1
inputs_coffee_bar
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
identity_11	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall!inputs_avg_cars_at_home_approx__1inputs_coffee_barinputs_floristinputs_low_fatinputs_num_children_at_homeinputs_prepared_foodinputs_salad_barinputs_store_sales_in_millions_inputs_store_sqftinputs_total_childreninputs_unit_sales_in_millions_inputs_video_storeunknown	unknown_0	unknown_1	unknown_2*
Tin
2*
Tout
2										*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *_
fZRX
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_257156o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0	*'
_output_shapes
:���������q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0	*'
_output_shapes
:���������q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:���������q

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0	*'
_output_shapes
:���������q

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0	*'
_output_shapes
:���������q

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:���������q

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:���������q

Identity_9Identity StatefulPartitionedCall:output:9^NoOp*
T0	*'
_output_shapes
:���������s
Identity_10Identity!StatefulPartitionedCall:output:10^NoOp*
T0	*'
_output_shapes
:���������s
Identity_11Identity!StatefulPartitionedCall:output:11^NoOp*
T0	*'
_output_shapes
:���������`
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
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
'
_output_shapes
:���������
;
_user_specified_name#!inputs_avg_cars_at_home_approx__1:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_coffee_bar:WS
'
_output_shapes
:���������
(
_user_specified_nameinputs_florist:WS
'
_output_shapes
:���������
(
_user_specified_nameinputs_low_fat:d`
'
_output_shapes
:���������
5
_user_specified_nameinputs_num_children_at_home:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs_prepared_food:YU
'
_output_shapes
:���������
*
_user_specified_nameinputs_salad_bar:hd
'
_output_shapes
:���������
9
_user_specified_name!inputs_store_sales_in_millions_:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_store_sqft:^	Z
'
_output_shapes
:���������
/
_user_specified_nameinputs_total_children:g
c
'
_output_shapes
:���������
8
_user_specified_name inputs_unit_sales_in_millions_:[W
'
_output_shapes
:���������
,
_user_specified_nameinputs_video_store:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
I__inference_concatenate_5_layer_call_and_return_conditional_losses_257883
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
value	B :�
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_8:Q	M
'
_output_shapes
:���������
"
_user_specified_name
inputs_9:R
N
'
_output_shapes
:���������
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_11
�

�
D__inference_dense_20_layer_call_and_return_conditional_losses_257903

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�0
�
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_257156

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
identity_11	��StatefulPartitionedCall;
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
valueB:�
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
valueB:�
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
:����������
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:���������*
dtype0	*
shape:����������
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1PlaceholderWithDefault:output:0inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11unknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2											*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *"
fR
__inference_pruned_256833o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:3^NoOp*
T0	*'
_output_shapes
:���������q

Identity_3Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:���������q

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0	*'
_output_shapes
:���������q

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0	*'
_output_shapes
:���������q

Identity_6Identity StatefulPartitionedCall:output:7^NoOp*
T0	*'
_output_shapes
:���������q

Identity_7Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:���������q

Identity_8Identity StatefulPartitionedCall:output:9^NoOp*
T0*'
_output_shapes
:���������r

Identity_9Identity!StatefulPartitionedCall:output:10^NoOp*
T0	*'
_output_shapes
:���������s
Identity_10Identity!StatefulPartitionedCall:output:11^NoOp*
T0	*'
_output_shapes
:���������s
Identity_11Identity!StatefulPartitionedCall:output:12^NoOp*
T0	*'
_output_shapes
:���������`
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
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:O	K
'
_output_shapes
:���������
 
_user_specified_nameinputs:O
K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
I__inference_concatenate_5_layer_call_and_return_conditional_losses_257343

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
value	B :�
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:O	K
'
_output_shapes
:���������
 
_user_specified_nameinputs:O
K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
$__inference_signature_wrapper_256878

inputs
inputs_1
	inputs_10
	inputs_11
	inputs_12
inputs_2	
inputs_3
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
identity_12	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12unknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2											*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *"
fR
__inference_pruned_256833`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0	*'
_output_shapes
:���������q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0	*'
_output_shapes
:���������q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:���������q

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0	*'
_output_shapes
:���������q

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0	*'
_output_shapes
:���������q

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0	*'
_output_shapes
:���������q

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:���������q

Identity_9Identity StatefulPartitionedCall:output:9^NoOp*
T0*'
_output_shapes
:���������s
Identity_10Identity!StatefulPartitionedCall:output:10^NoOp*
T0	*'
_output_shapes
:���������s
Identity_11Identity!StatefulPartitionedCall:output:11^NoOp*
T0	*'
_output_shapes
:���������s
Identity_12Identity!StatefulPartitionedCall:output:12^NoOp*
T0	*'
_output_shapes
:���������"
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
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_11:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_12:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_5:Q	M
'
_output_shapes
:���������
"
_user_specified_name
inputs_6:Q
M
'
_output_shapes
:���������
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_8:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_9:
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
�
�
(__inference_model_5_layer_call_fn_257760(
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
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_257571o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:m i
'
_output_shapes
:���������
>
_user_specified_name&$inputs_avg_cars_at_home_approx__1_xf:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs_coffee_bar_xf:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_florist_xf:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs_low_fat_xf:gc
'
_output_shapes
:���������
8
_user_specified_name inputs_num_children_at_home_xf:`\
'
_output_shapes
:���������
1
_user_specified_nameinputs_prepared_food_xf:\X
'
_output_shapes
:���������
-
_user_specified_nameinputs_salad_bar_xf:kg
'
_output_shapes
:���������
<
_user_specified_name$"inputs_store_sales_in_millions__xf:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs_store_sqft_xf:a	]
'
_output_shapes
:���������
2
_user_specified_nameinputs_total_children_xf:j
f
'
_output_shapes
:���������
;
_user_specified_name#!inputs_unit_sales_in_millions__xf:^Z
'
_output_shapes
:���������
/
_user_specified_nameinputs_video_store_xf
�$
�
C__inference_model_5_layer_call_and_return_conditional_losses_257414

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
dense_20_257357:	�
dense_20_257359:	�"
dense_21_257374:	�
dense_21_257376:!
dense_22_257391:
dense_22_257393:!
dense_23_257408:
dense_23_257410:
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�
concatenate_5/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_257343�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_20_257357dense_20_257359*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_257356�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_257374dense_21_257376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_257373�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_257391dense_22_257393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_257390�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_257408dense_23_257410*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_257407x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:O	K
'
_output_shapes
:���������
 
_user_specified_nameinputs:O
K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_model_5_layer_call_fn_257433
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
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallplaceholdercoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfplaceholder_1store_sqft_xftotal_children_xfplaceholder_2video_store_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_257414o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
'
_output_shapes
:���������
7
_user_specified_nameavg_cars_at home(approx).1_xf:VR
'
_output_shapes
:���������
'
_user_specified_namecoffee_bar_xf:SO
'
_output_shapes
:���������
$
_user_specified_name
florist_xf:SO
'
_output_shapes
:���������
$
_user_specified_name
low_fat_xf:`\
'
_output_shapes
:���������
1
_user_specified_namenum_children_at_home_xf:YU
'
_output_shapes
:���������
*
_user_specified_nameprepared_food_xf:UQ
'
_output_shapes
:���������
&
_user_specified_namesalad_bar_xf:d`
'
_output_shapes
:���������
5
_user_specified_namestore_sales(in millions)_xf:VR
'
_output_shapes
:���������
'
_user_specified_namestore_sqft_xf:Z	V
'
_output_shapes
:���������
+
_user_specified_nametotal_children_xf:c
_
'
_output_shapes
:���������
4
_user_specified_nameunit_sales(in millions)_xf:WS
'
_output_shapes
:���������
(
_user_specified_namevideo_store_xf"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
examples-
serving_default_examples:0���������<
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
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
layer_with_weights-2
layer-15
layer_with_weights-3
layer-16
layer-17
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	tft_layer

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
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

0kernel
1bias"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias"
_tf_keras_layer
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
$H _saved_model_loader_tracked_dict"
_tf_keras_model
X
(0
)1
02
13
84
95
@6
A7"
trackable_list_wrapper
X
(0
)1
02
13
84
95
@6
A7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ntrace_0
Otrace_1
Ptrace_2
Qtrace_32�
(__inference_model_5_layer_call_fn_257433
(__inference_model_5_layer_call_fn_257728
(__inference_model_5_layer_call_fn_257760
(__inference_model_5_layer_call_fn_257622�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zNtrace_0zOtrace_1zPtrace_2zQtrace_3
�
Rtrace_0
Strace_1
Ttrace_2
Utrace_32�
C__inference_model_5_layer_call_and_return_conditional_losses_257805
C__inference_model_5_layer_call_and_return_conditional_losses_257850
C__inference_model_5_layer_call_and_return_conditional_losses_257658
C__inference_model_5_layer_call_and_return_conditional_losses_257694�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zRtrace_0zStrace_1zTtrace_2zUtrace_3
�B�
!__inference__wrapped_model_257079avg_cars_at home(approx).1_xfcoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfstore_sales(in millions)_xfstore_sqft_xftotal_children_xfunit_sales(in millions)_xfvideo_store_xf"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
V
_variables
W_iterations
X_learning_rate
Y_index_dict
Z
_momentums
[_velocities
\_update_step_xla"
experimentalOptimizer
,
]serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
ctrace_02�
.__inference_concatenate_5_layer_call_fn_257866�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zctrace_0
�
dtrace_02�
I__inference_concatenate_5_layer_call_and_return_conditional_losses_257883�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
jtrace_02�
)__inference_dense_20_layer_call_fn_257892�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zjtrace_0
�
ktrace_02�
D__inference_dense_20_layer_call_and_return_conditional_losses_257903�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zktrace_0
": 	�2dense_20/kernel
:�2dense_20/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
qtrace_02�
)__inference_dense_21_layer_call_fn_257912�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zqtrace_0
�
rtrace_02�
D__inference_dense_21_layer_call_and_return_conditional_losses_257923�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zrtrace_0
": 	�2dense_21/kernel
:2dense_21/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
xtrace_02�
)__inference_dense_22_layer_call_fn_257932�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zxtrace_0
�
ytrace_02�
D__inference_dense_22_layer_call_and_return_conditional_losses_257943�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0
!:2dense_22/kernel
:2dense_22/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
trace_02�
)__inference_dense_23_layer_call_fn_257952�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
�trace_02�
D__inference_dense_23_layer_call_and_return_conditional_losses_257963�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:2dense_23/kernel
:2dense_23/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
;__inference_transform_features_layer_1_layer_call_fn_257189
;__inference_transform_features_layer_1_layer_call_fn_258009�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_258071
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_257297�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	_imported
�_wrapped_function
�_structured_inputs
�_structured_outputs
�_output_to_inputs_map"
trackable_dict_wrapper
 "
trackable_list_wrapper
�
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
15
16
17"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_5_layer_call_fn_257433avg_cars_at home(approx).1_xfcoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfstore_sales(in millions)_xfstore_sqft_xftotal_children_xfunit_sales(in millions)_xfvideo_store_xf"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_5_layer_call_fn_257728$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xf"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_5_layer_call_fn_257760$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xf"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_5_layer_call_fn_257622avg_cars_at home(approx).1_xfcoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfstore_sales(in millions)_xfstore_sqft_xftotal_children_xfunit_sales(in millions)_xfvideo_store_xf"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_5_layer_call_and_return_conditional_losses_257805$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xf"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_5_layer_call_and_return_conditional_losses_257850$inputs_avg_cars_at_home_approx__1_xfinputs_coffee_bar_xfinputs_florist_xfinputs_low_fat_xfinputs_num_children_at_home_xfinputs_prepared_food_xfinputs_salad_bar_xf"inputs_store_sales_in_millions__xfinputs_store_sqft_xfinputs_total_children_xf!inputs_unit_sales_in_millions__xfinputs_video_store_xf"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_5_layer_call_and_return_conditional_losses_257658avg_cars_at home(approx).1_xfcoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfstore_sales(in millions)_xfstore_sqft_xftotal_children_xfunit_sales(in millions)_xfvideo_store_xf"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_5_layer_call_and_return_conditional_losses_257694avg_cars_at home(approx).1_xfcoffee_bar_xf
florist_xf
low_fat_xfnum_children_at_home_xfprepared_food_xfsalad_bar_xfstore_sales(in millions)_xfstore_sqft_xftotal_children_xfunit_sales(in millions)_xfvideo_store_xf"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
W0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
�2��
���
FullArgSpec2
args*�'
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3B�
$__inference_signature_wrapper_257033examples"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3
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
�B�
.__inference_concatenate_5_layer_call_fn_257866inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_concatenate_5_layer_call_and_return_conditional_losses_257883inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_dense_20_layer_call_fn_257892inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_20_layer_call_and_return_conditional_losses_257903inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_dense_21_layer_call_fn_257912inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_21_layer_call_and_return_conditional_losses_257923inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_dense_22_layer_call_fn_257932inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_22_layer_call_and_return_conditional_losses_257943inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_dense_23_layer_call_fn_257952inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_23_layer_call_and_return_conditional_losses_257963inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3B�
;__inference_transform_features_layer_1_layer_call_fn_257189avg_cars_at home(approx).1
coffee_barfloristlow_fatnum_children_at_homeprepared_food	salad_barstore_sales(in millions)
store_sqfttotal_childrenunit_sales(in millions)video_store"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3B�
;__inference_transform_features_layer_1_layer_call_fn_258009!inputs_avg_cars_at_home_approx__1inputs_coffee_barinputs_floristinputs_low_fatinputs_num_children_at_homeinputs_prepared_foodinputs_salad_barinputs_store_sales_in_millions_inputs_store_sqftinputs_total_childreninputs_unit_sales_in_millions_inputs_video_store"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3B�
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_258071!inputs_avg_cars_at_home_approx__1inputs_coffee_barinputs_floristinputs_low_fatinputs_num_children_at_homeinputs_prepared_foodinputs_salad_barinputs_store_sales_in_millions_inputs_store_sqftinputs_total_childreninputs_unit_sales_in_millions_inputs_video_store"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3B�
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_257297avg_cars_at home(approx).1
coffee_barfloristlow_fatnum_children_at_homeprepared_food	salad_barstore_sales(in millions)
store_sqfttotal_childrenunit_sales(in millions)video_store"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3
�
�created_variables
�	resources
�trackable_objects
�initializers
�assets
�
signatures
$�_self_saveable_object_factories
�transform_fn"
_generic_user_object
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3B�
__inference_pruned_256833inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12z�	capture_0z�	capture_1z�	capture_2z�	capture_3
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
':%	�2Adam/m/dense_20/kernel
':%	�2Adam/v/dense_20/kernel
!:�2Adam/m/dense_20/bias
!:�2Adam/v/dense_20/bias
':%	�2Adam/m/dense_21/kernel
':%	�2Adam/v/dense_21/kernel
 :2Adam/m/dense_21/bias
 :2Adam/v/dense_21/bias
&:$2Adam/m/dense_22/kernel
&:$2Adam/v/dense_22/kernel
 :2Adam/m/dense_22/bias
 :2Adam/v/dense_22/bias
&:$2Adam/m/dense_23/kernel
&:$2Adam/v/dense_23/kernel
 :2Adam/m/dense_23/bias
 :2Adam/v/dense_23/bias
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
�serving_default"
signature_map
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3B�
$__inference_signature_wrapper_256878inputsinputs_1	inputs_10	inputs_11	inputs_12inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3�
!__inference__wrapped_model_257079�()0189@A���
���
���
X
avg_cars_at home(approx).1_xf7�4
avg_cars_at home(approx).1_xf���������
8
coffee_bar_xf'�$
coffee_bar_xf���������
2

florist_xf$�!

florist_xf���������
2

low_fat_xf$�!

low_fat_xf���������
L
num_children_at_home_xf1�.
num_children_at_home_xf���������
>
prepared_food_xf*�'
prepared_food_xf���������
6
salad_bar_xf&�#
salad_bar_xf���������
T
store_sales(in millions)_xf5�2
store_sales(in millions)_xf���������
8
store_sqft_xf'�$
store_sqft_xf���������
@
total_children_xf+�(
total_children_xf���������
R
unit_sales(in millions)_xf4�1
unit_sales(in millions)_xf���������
:
video_store_xf(�%
video_store_xf���������
� "3�0
.
dense_23"�
dense_23����������
I__inference_concatenate_5_layer_call_and_return_conditional_losses_257883����
���
���
"�
inputs_0���������
"�
inputs_1���������
"�
inputs_2���������
"�
inputs_3���������
"�
inputs_4���������
"�
inputs_5���������
"�
inputs_6���������
"�
inputs_7���������
"�
inputs_8���������
"�
inputs_9���������
#� 
	inputs_10���������
#� 
	inputs_11���������
� ",�)
"�
tensor_0���������
� �
.__inference_concatenate_5_layer_call_fn_257866����
���
���
"�
inputs_0���������
"�
inputs_1���������
"�
inputs_2���������
"�
inputs_3���������
"�
inputs_4���������
"�
inputs_5���������
"�
inputs_6���������
"�
inputs_7���������
"�
inputs_8���������
"�
inputs_9���������
#� 
	inputs_10���������
#� 
	inputs_11���������
� "!�
unknown����������
D__inference_dense_20_layer_call_and_return_conditional_losses_257903d()/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
)__inference_dense_20_layer_call_fn_257892Y()/�,
%�"
 �
inputs���������
� ""�
unknown�����������
D__inference_dense_21_layer_call_and_return_conditional_losses_257923d010�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_21_layer_call_fn_257912Y010�-
&�#
!�
inputs����������
� "!�
unknown����������
D__inference_dense_22_layer_call_and_return_conditional_losses_257943c89/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_22_layer_call_fn_257932X89/�,
%�"
 �
inputs���������
� "!�
unknown����������
D__inference_dense_23_layer_call_and_return_conditional_losses_257963c@A/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_23_layer_call_fn_257952X@A/�,
%�"
 �
inputs���������
� "!�
unknown����������
C__inference_model_5_layer_call_and_return_conditional_losses_257658�()0189@A���
���
���
X
avg_cars_at home(approx).1_xf7�4
avg_cars_at home(approx).1_xf���������
8
coffee_bar_xf'�$
coffee_bar_xf���������
2

florist_xf$�!

florist_xf���������
2

low_fat_xf$�!

low_fat_xf���������
L
num_children_at_home_xf1�.
num_children_at_home_xf���������
>
prepared_food_xf*�'
prepared_food_xf���������
6
salad_bar_xf&�#
salad_bar_xf���������
T
store_sales(in millions)_xf5�2
store_sales(in millions)_xf���������
8
store_sqft_xf'�$
store_sqft_xf���������
@
total_children_xf+�(
total_children_xf���������
R
unit_sales(in millions)_xf4�1
unit_sales(in millions)_xf���������
:
video_store_xf(�%
video_store_xf���������
p 

 
� ",�)
"�
tensor_0���������
� �
C__inference_model_5_layer_call_and_return_conditional_losses_257694�()0189@A���
���
���
X
avg_cars_at home(approx).1_xf7�4
avg_cars_at home(approx).1_xf���������
8
coffee_bar_xf'�$
coffee_bar_xf���������
2

florist_xf$�!

florist_xf���������
2

low_fat_xf$�!

low_fat_xf���������
L
num_children_at_home_xf1�.
num_children_at_home_xf���������
>
prepared_food_xf*�'
prepared_food_xf���������
6
salad_bar_xf&�#
salad_bar_xf���������
T
store_sales(in millions)_xf5�2
store_sales(in millions)_xf���������
8
store_sqft_xf'�$
store_sqft_xf���������
@
total_children_xf+�(
total_children_xf���������
R
unit_sales(in millions)_xf4�1
unit_sales(in millions)_xf���������
:
video_store_xf(�%
video_store_xf���������
p

 
� ",�)
"�
tensor_0���������
� �
C__inference_model_5_layer_call_and_return_conditional_losses_257805�()0189@A���
���
���
_
avg_cars_at home(approx).1_xf>�;
$inputs_avg_cars_at_home_approx__1_xf���������
?
coffee_bar_xf.�+
inputs_coffee_bar_xf���������
9

florist_xf+�(
inputs_florist_xf���������
9

low_fat_xf+�(
inputs_low_fat_xf���������
S
num_children_at_home_xf8�5
inputs_num_children_at_home_xf���������
E
prepared_food_xf1�.
inputs_prepared_food_xf���������
=
salad_bar_xf-�*
inputs_salad_bar_xf���������
[
store_sales(in millions)_xf<�9
"inputs_store_sales_in_millions__xf���������
?
store_sqft_xf.�+
inputs_store_sqft_xf���������
G
total_children_xf2�/
inputs_total_children_xf���������
Y
unit_sales(in millions)_xf;�8
!inputs_unit_sales_in_millions__xf���������
A
video_store_xf/�,
inputs_video_store_xf���������
p 

 
� ",�)
"�
tensor_0���������
� �
C__inference_model_5_layer_call_and_return_conditional_losses_257850�()0189@A���
���
���
_
avg_cars_at home(approx).1_xf>�;
$inputs_avg_cars_at_home_approx__1_xf���������
?
coffee_bar_xf.�+
inputs_coffee_bar_xf���������
9

florist_xf+�(
inputs_florist_xf���������
9

low_fat_xf+�(
inputs_low_fat_xf���������
S
num_children_at_home_xf8�5
inputs_num_children_at_home_xf���������
E
prepared_food_xf1�.
inputs_prepared_food_xf���������
=
salad_bar_xf-�*
inputs_salad_bar_xf���������
[
store_sales(in millions)_xf<�9
"inputs_store_sales_in_millions__xf���������
?
store_sqft_xf.�+
inputs_store_sqft_xf���������
G
total_children_xf2�/
inputs_total_children_xf���������
Y
unit_sales(in millions)_xf;�8
!inputs_unit_sales_in_millions__xf���������
A
video_store_xf/�,
inputs_video_store_xf���������
p

 
� ",�)
"�
tensor_0���������
� �
(__inference_model_5_layer_call_fn_257433�()0189@A���
���
���
X
avg_cars_at home(approx).1_xf7�4
avg_cars_at home(approx).1_xf���������
8
coffee_bar_xf'�$
coffee_bar_xf���������
2

florist_xf$�!

florist_xf���������
2

low_fat_xf$�!

low_fat_xf���������
L
num_children_at_home_xf1�.
num_children_at_home_xf���������
>
prepared_food_xf*�'
prepared_food_xf���������
6
salad_bar_xf&�#
salad_bar_xf���������
T
store_sales(in millions)_xf5�2
store_sales(in millions)_xf���������
8
store_sqft_xf'�$
store_sqft_xf���������
@
total_children_xf+�(
total_children_xf���������
R
unit_sales(in millions)_xf4�1
unit_sales(in millions)_xf���������
:
video_store_xf(�%
video_store_xf���������
p 

 
� "!�
unknown����������
(__inference_model_5_layer_call_fn_257622�()0189@A���
���
���
X
avg_cars_at home(approx).1_xf7�4
avg_cars_at home(approx).1_xf���������
8
coffee_bar_xf'�$
coffee_bar_xf���������
2

florist_xf$�!

florist_xf���������
2

low_fat_xf$�!

low_fat_xf���������
L
num_children_at_home_xf1�.
num_children_at_home_xf���������
>
prepared_food_xf*�'
prepared_food_xf���������
6
salad_bar_xf&�#
salad_bar_xf���������
T
store_sales(in millions)_xf5�2
store_sales(in millions)_xf���������
8
store_sqft_xf'�$
store_sqft_xf���������
@
total_children_xf+�(
total_children_xf���������
R
unit_sales(in millions)_xf4�1
unit_sales(in millions)_xf���������
:
video_store_xf(�%
video_store_xf���������
p

 
� "!�
unknown����������
(__inference_model_5_layer_call_fn_257728�()0189@A���
���
���
_
avg_cars_at home(approx).1_xf>�;
$inputs_avg_cars_at_home_approx__1_xf���������
?
coffee_bar_xf.�+
inputs_coffee_bar_xf���������
9

florist_xf+�(
inputs_florist_xf���������
9

low_fat_xf+�(
inputs_low_fat_xf���������
S
num_children_at_home_xf8�5
inputs_num_children_at_home_xf���������
E
prepared_food_xf1�.
inputs_prepared_food_xf���������
=
salad_bar_xf-�*
inputs_salad_bar_xf���������
[
store_sales(in millions)_xf<�9
"inputs_store_sales_in_millions__xf���������
?
store_sqft_xf.�+
inputs_store_sqft_xf���������
G
total_children_xf2�/
inputs_total_children_xf���������
Y
unit_sales(in millions)_xf;�8
!inputs_unit_sales_in_millions__xf���������
A
video_store_xf/�,
inputs_video_store_xf���������
p 

 
� "!�
unknown����������
(__inference_model_5_layer_call_fn_257760�()0189@A���
���
���
_
avg_cars_at home(approx).1_xf>�;
$inputs_avg_cars_at_home_approx__1_xf���������
?
coffee_bar_xf.�+
inputs_coffee_bar_xf���������
9

florist_xf+�(
inputs_florist_xf���������
9

low_fat_xf+�(
inputs_low_fat_xf���������
S
num_children_at_home_xf8�5
inputs_num_children_at_home_xf���������
E
prepared_food_xf1�.
inputs_prepared_food_xf���������
=
salad_bar_xf-�*
inputs_salad_bar_xf���������
[
store_sales(in millions)_xf<�9
"inputs_store_sales_in_millions__xf���������
?
store_sqft_xf.�+
inputs_store_sqft_xf���������
G
total_children_xf2�/
inputs_total_children_xf���������
Y
unit_sales(in millions)_xf;�8
!inputs_unit_sales_in_millions__xf���������
A
video_store_xf/�,
inputs_video_store_xf���������
p

 
� "!�
unknown����������
__inference_pruned_256833��������
���
���
Y
avg_cars_at home(approx).1;�8
!inputs_avg_cars_at_home_approx__1���������
9

coffee_bar+�(
inputs_coffee_bar���������
5
cost_bin)�&
inputs_cost_bin���������	
3
florist(�%
inputs_florist���������
3
low_fat(�%
inputs_low_fat���������
M
num_children_at_home5�2
inputs_num_children_at_home���������
?
prepared_food.�+
inputs_prepared_food���������
7
	salad_bar*�'
inputs_salad_bar���������
U
store_sales(in millions)9�6
inputs_store_sales_in_millions_���������
9

store_sqft+�(
inputs_store_sqft���������
A
total_children/�,
inputs_total_children���������
S
unit_sales(in millions)8�5
inputs_unit_sales_in_millions_���������
;
video_store,�)
inputs_video_store���������
� "���
X
avg_cars_at home(approx).1_xf7�4
avg_cars_at_home_approx__1_xf���������	
8
coffee_bar_xf'�$
coffee_bar_xf���������	
4
cost_bin_xf%�"
cost_bin_xf���������	
2

florist_xf$�!

florist_xf���������	
2

low_fat_xf$�!

low_fat_xf���������	
L
num_children_at_home_xf1�.
num_children_at_home_xf���������	
>
prepared_food_xf*�'
prepared_food_xf���������	
6
salad_bar_xf&�#
salad_bar_xf���������	
T
store_sales(in millions)_xf5�2
store_sales_in_millions__xf���������
8
store_sqft_xf'�$
store_sqft_xf���������
@
total_children_xf+�(
total_children_xf���������	
R
unit_sales(in millions)_xf4�1
unit_sales_in_millions__xf���������	
:
video_store_xf(�%
video_store_xf���������	�
$__inference_signature_wrapper_256878��������
� 
���
*
inputs �
inputs���������
.
inputs_1"�
inputs_1���������
0
	inputs_10#� 
	inputs_10���������
0
	inputs_11#� 
	inputs_11���������
0
	inputs_12#� 
	inputs_12���������
.
inputs_2"�
inputs_2���������	
.
inputs_3"�
inputs_3���������
.
inputs_4"�
inputs_4���������
.
inputs_5"�
inputs_5���������
.
inputs_6"�
inputs_6���������
.
inputs_7"�
inputs_7���������
.
inputs_8"�
inputs_8���������
.
inputs_9"�
inputs_9���������"���
X
avg_cars_at home(approx).1_xf7�4
avg_cars_at_home_approx__1_xf���������	
8
coffee_bar_xf'�$
coffee_bar_xf���������	
4
cost_bin_xf%�"
cost_bin_xf���������	
2

florist_xf$�!

florist_xf���������	
2

low_fat_xf$�!

low_fat_xf���������	
L
num_children_at_home_xf1�.
num_children_at_home_xf���������	
>
prepared_food_xf*�'
prepared_food_xf���������	
6
salad_bar_xf&�#
salad_bar_xf���������	
T
store_sales(in millions)_xf5�2
store_sales_in_millions__xf���������
8
store_sqft_xf'�$
store_sqft_xf���������
@
total_children_xf+�(
total_children_xf���������	
R
unit_sales(in millions)_xf4�1
unit_sales_in_millions__xf���������	
:
video_store_xf(�%
video_store_xf���������	�
$__inference_signature_wrapper_257033�����()0189@A9�6
� 
/�,
*
examples�
examples���������"3�0
.
output_0"�
output_0����������
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_257297��������
���
���
R
avg_cars_at home(approx).14�1
avg_cars_at home(approx).1���������
2

coffee_bar$�!

coffee_bar���������
,
florist!�
florist���������
,
low_fat!�
low_fat���������
F
num_children_at_home.�+
num_children_at_home���������
8
prepared_food'�$
prepared_food���������
0
	salad_bar#� 
	salad_bar���������
N
store_sales(in millions)2�/
store_sales(in millions)���������
2

store_sqft$�!

store_sqft���������
:
total_children(�%
total_children���������
L
unit_sales(in millions)1�.
unit_sales(in millions)���������
4
video_store%�"
video_store���������
� "���
���
a
avg_cars_at home(approx).1_xf@�=
&tensor_0_avg_cars_at_home_approx__1_xf���������	
A
coffee_bar_xf0�-
tensor_0_coffee_bar_xf���������	
;

florist_xf-�*
tensor_0_florist_xf���������	
;

low_fat_xf-�*
tensor_0_low_fat_xf���������	
U
num_children_at_home_xf:�7
 tensor_0_num_children_at_home_xf���������	
G
prepared_food_xf3�0
tensor_0_prepared_food_xf���������	
?
salad_bar_xf/�,
tensor_0_salad_bar_xf���������	
]
store_sales(in millions)_xf>�;
$tensor_0_store_sales_in_millions__xf���������
A
store_sqft_xf0�-
tensor_0_store_sqft_xf���������
I
total_children_xf4�1
tensor_0_total_children_xf���������	
[
unit_sales(in millions)_xf=�:
#tensor_0_unit_sales_in_millions__xf���������	
C
video_store_xf1�.
tensor_0_video_store_xf���������	
� �
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_258071��������
���
���
Y
avg_cars_at home(approx).1;�8
!inputs_avg_cars_at_home_approx__1���������
9

coffee_bar+�(
inputs_coffee_bar���������
3
florist(�%
inputs_florist���������
3
low_fat(�%
inputs_low_fat���������
M
num_children_at_home5�2
inputs_num_children_at_home���������
?
prepared_food.�+
inputs_prepared_food���������
7
	salad_bar*�'
inputs_salad_bar���������
U
store_sales(in millions)9�6
inputs_store_sales_in_millions_���������
9

store_sqft+�(
inputs_store_sqft���������
A
total_children/�,
inputs_total_children���������
S
unit_sales(in millions)8�5
inputs_unit_sales_in_millions_���������
;
video_store,�)
inputs_video_store���������
� "���
���
a
avg_cars_at home(approx).1_xf@�=
&tensor_0_avg_cars_at_home_approx__1_xf���������	
A
coffee_bar_xf0�-
tensor_0_coffee_bar_xf���������	
;

florist_xf-�*
tensor_0_florist_xf���������	
;

low_fat_xf-�*
tensor_0_low_fat_xf���������	
U
num_children_at_home_xf:�7
 tensor_0_num_children_at_home_xf���������	
G
prepared_food_xf3�0
tensor_0_prepared_food_xf���������	
?
salad_bar_xf/�,
tensor_0_salad_bar_xf���������	
]
store_sales(in millions)_xf>�;
$tensor_0_store_sales_in_millions__xf���������
A
store_sqft_xf0�-
tensor_0_store_sqft_xf���������
I
total_children_xf4�1
tensor_0_total_children_xf���������	
[
unit_sales(in millions)_xf=�:
#tensor_0_unit_sales_in_millions__xf���������	
C
video_store_xf1�.
tensor_0_video_store_xf���������	
� �
;__inference_transform_features_layer_1_layer_call_fn_257189��������
���
���
R
avg_cars_at home(approx).14�1
avg_cars_at home(approx).1���������
2

coffee_bar$�!

coffee_bar���������
,
florist!�
florist���������
,
low_fat!�
low_fat���������
F
num_children_at_home.�+
num_children_at_home���������
8
prepared_food'�$
prepared_food���������
0
	salad_bar#� 
	salad_bar���������
N
store_sales(in millions)2�/
store_sales(in millions)���������
2

store_sqft$�!

store_sqft���������
:
total_children(�%
total_children���������
L
unit_sales(in millions)1�.
unit_sales(in millions)���������
4
video_store%�"
video_store���������
� "���
X
avg_cars_at home(approx).1_xf7�4
avg_cars_at_home_approx__1_xf���������	
8
coffee_bar_xf'�$
coffee_bar_xf���������	
2

florist_xf$�!

florist_xf���������	
2

low_fat_xf$�!

low_fat_xf���������	
L
num_children_at_home_xf1�.
num_children_at_home_xf���������	
>
prepared_food_xf*�'
prepared_food_xf���������	
6
salad_bar_xf&�#
salad_bar_xf���������	
T
store_sales(in millions)_xf5�2
store_sales_in_millions__xf���������
8
store_sqft_xf'�$
store_sqft_xf���������
@
total_children_xf+�(
total_children_xf���������	
R
unit_sales(in millions)_xf4�1
unit_sales_in_millions__xf���������	
:
video_store_xf(�%
video_store_xf���������	�
;__inference_transform_features_layer_1_layer_call_fn_258009��������
���
���
Y
avg_cars_at home(approx).1;�8
!inputs_avg_cars_at_home_approx__1���������
9

coffee_bar+�(
inputs_coffee_bar���������
3
florist(�%
inputs_florist���������
3
low_fat(�%
inputs_low_fat���������
M
num_children_at_home5�2
inputs_num_children_at_home���������
?
prepared_food.�+
inputs_prepared_food���������
7
	salad_bar*�'
inputs_salad_bar���������
U
store_sales(in millions)9�6
inputs_store_sales_in_millions_���������
9

store_sqft+�(
inputs_store_sqft���������
A
total_children/�,
inputs_total_children���������
S
unit_sales(in millions)8�5
inputs_unit_sales_in_millions_���������
;
video_store,�)
inputs_video_store���������
� "���
X
avg_cars_at home(approx).1_xf7�4
avg_cars_at_home_approx__1_xf���������	
8
coffee_bar_xf'�$
coffee_bar_xf���������	
2

florist_xf$�!

florist_xf���������	
2

low_fat_xf$�!

low_fat_xf���������	
L
num_children_at_home_xf1�.
num_children_at_home_xf���������	
>
prepared_food_xf*�'
prepared_food_xf���������	
6
salad_bar_xf&�#
salad_bar_xf���������	
T
store_sales(in millions)_xf5�2
store_sales_in_millions__xf���������
8
store_sqft_xf'�$
store_sqft_xf���������
@
total_children_xf+�(
total_children_xf���������	
R
unit_sales(in millions)_xf4�1
unit_sales_in_millions__xf���������	
:
video_store_xf(�%
video_store_xf���������	