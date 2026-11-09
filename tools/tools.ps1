# Define an array containing all the sub-optical field sizes to test
python tools/pose_val.py --dataset mouse --model-type spipose --models n --imgsz 640 --seed 20250221 --split test --save-json --seed 20250221 --optical-field-sizes 128 --sub-optical-field-sizes 64 --imgsz-hadamard 256 --aliasing --save-json
python tools/pose_val.py --dataset mouse --model-type spipose --models n --imgsz 640 --seed 20250221 --split test --save-json --seed 20250221 --optical-field-sizes 128 --sub-optical-field-sizes 32 --imgsz-hadamard 256 --aliasing --save-json
python tools/pose_val.py --dataset mouse --model-type spipose --models n --imgsz 640 --seed 20250221 --split test --save-json --seed 20250221 --optical-field-sizes 128 --sub-optical-field-sizes 16 --imgsz-hadamard 256 --aliasing --save-json

python tools/pose_val.py --dataset mouse --model-type spipose --models s,m,l,x --imgsz 256 --seed 20250221 --split test --save-json --seed 20250221 --optical-field-sizes 128 --sub-optical-field-sizes 64 --imgsz-hadamard 256 --aliasing --save-json
python tools/pose_val.py --dataset mouse --model-type spipose --models s,m,l,x --imgsz 256 --seed 20250221 --split test --save-json --seed 20250221 --optical-field-sizes 128 --sub-optical-field-sizes 32 --imgsz-hadamard 256 --aliasing --save-json
python tools/pose_val.py --dataset mouse --model-type spipose --models s,m,l,x --imgsz 256 --seed 20250221 --split test --save-json --seed 20250221 --optical-field-sizes 128 --sub-optical-field-sizes 16 --imgsz-hadamard 256 --aliasing --save-json

python tools/pose_val.py --dataset twoperson --model-type spipose --models n,s,m,l,x --imgsz 640 --seed 20250221 --split test --save-json --seed 20250221 --optical-field-sizes 128 --sub-optical-field-sizes 64 --imgsz-hadamard 256 --aliasing --save-json
python tools/pose_val.py --dataset twoperson --model-type spipose --models n,s,m,l,x --imgsz 640 --seed 20250221 --split test --save-json --seed 20250221 --optical-field-sizes 128 --sub-optical-field-sizes 32 --imgsz-hadamard 256 --aliasing --save-json
python tools/pose_val.py --dataset twoperson --model-type spipose --models n,s,m,l,x --imgsz 640 --seed 20250221 --split test --save-json --seed 20250221 --optical-field-sizes 128 --sub-optical-field-sizes 16 --imgsz-hadamard 256 --aliasing --save-json

python tools/pose_val.py --dataset fly --model-type spipose --models n,s,m,l,x --imgsz 192 --seed 20250221 --split test --save-json --seed 20250221 --optical-field-sizes 128 --sub-optical-field-sizes 64 --imgsz-hadamard 256 --aliasing --save-json
python tools/pose_val.py --dataset fly --model-type spipose --models n,s,m,l,x --imgsz 192 --seed 20250221 --split test --save-json --seed 20250221 --optical-field-sizes 128 --sub-optical-field-sizes 32 --imgsz-hadamard 256 --aliasing --save-json
python tools/pose_val.py --dataset fly --model-type spipose --models n,s,m,l,x --imgsz 192 --seed 20250221 --split test --save-json --seed 20250221 --optical-field-sizes 128 --sub-optical-field-sizes 16 --imgsz-hadamard 256 --aliasing --save-json


