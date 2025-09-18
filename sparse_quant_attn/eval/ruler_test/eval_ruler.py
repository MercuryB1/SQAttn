import os
from sparse_quant_attn.eval.ruler.data.prepare import prepare_data
from sparse_quant_attn.eval.ruler.pred.call_api import call_api


def eval_ruler(root_dir, model_name, method, bit8_thres, bit4_thres):
    seq_length = [
        4096,
        8192, 
        16384,
        32768,
        65536,
        131072
    ]

    tasks = [
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multivalue",
        "niah_multiquery",
        "vt",
        "cwe",
        "fwe",
        "qa_1",
        "qa_2"
    ]
    
    benchmark="synthetic"
    for max_seq_length in seq_length:
        result_dir = f"{root_dir}/{model_name}/{method}/{bit8_thres}_{bit4_thres}/{benchmark}/{max_seq_length}"
        data_dir = result_dir + '/data'
        pred_dir = result_dir + '/pred'
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        for task in task:
            prepare_data(
                task=task, 
                tokenizer_path=model_name,
                max_seq_length=max_seq_length,
                save_dir=data_dir,
                benchmark=benchmark,
                model_template_type='llama-3',
                tokenizer_type='hf',
                num_samples=25,
                remove_newline_tab=True
            )
            call_api(
                
            )
            