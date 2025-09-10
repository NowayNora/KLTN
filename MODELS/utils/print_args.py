import yaml

def print_args(args, save_path=None):
    def bold(text): return f"\033[1m{text}\033[0m"

    config = {
        "Basic Configuration": {
            "Task Name": args.task_name,
            "Is Training": args.is_training,
            "Model ID": args.model_id,
            "Model Name": args.model
        },
        "Data Configuration": {
            "Data": args.data,
            "Root Path": args.root_path,
            "Data Path": args.data_path,
            "Features": args.features,
            "Target": args.target,
            "Frequency": args.freq,
            "Checkpoints": args.checkpoints
        },
        "Model Parameters": {
            "Top K": args.top_k,
            "Num Kernels": args.num_kernels,
            "Encoder Input": args.enc_in,
            "Decoder Input": args.dec_in,
            "Output Channels": args.c_out,
            "Model Dim": args.d_model,
            "Heads": args.n_heads,
            "Encoder Layers": args.e_layers,
            "Decoder Layers": args.d_layers,
            "Feedforward Dim": args.d_ff,
            "Moving Avg": args.moving_avg,
            "Factor": args.factor,
            "Distillation": args.distil,
            "Dropout": args.dropout,
            "Embedding": args.embed,
            "Activation": args.activation
        },
        "Training Parameters": {
            "Workers": args.num_workers,
            "Iterations": args.itr,
            "Train Epochs": args.train_epochs,
            "Batch Size": args.batch_size,
            "Patience": args.patience,
            "Learning Rate": args.learning_rate,
            "Description": args.des,
            "Loss": args.loss,
            "LR Adjust": args.lradj,
            "Use AMP": args.use_amp
        },
        "GPU Configuration": {
            "Use GPU": args.use_gpu,
            "GPU": args.gpu,
            "Multi GPU": args.use_multi_gpu,
            "Devices": args.devices
        },
        "De-Stationary Projector Parameters": {
            "Hidden Dims": args.p_hidden_dims,
            "Hidden Layers": args.p_hidden_layers
        }
    }

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        config["Forecasting Parameters"] = {
            "Sequence Length": args.seq_len,
            "Label Length": args.label_len,
            "Prediction Length": args.pred_len,
            "Seasonality": args.seasonal_patterns,
            "Inverse Transform": args.inverse
        }

    if args.task_name == 'imputation':
        config["Imputation Parameters"] = {
            "Mask Rate": args.mask_rate
        }

    if args.task_name == 'anomaly_detection':
        config["Anomaly Detection Parameters"] = {
            "Anomaly Ratio": args.anomaly_ratio
        }

    # Print
    for section, params in config.items():
        print(bold(f">>> {section}"))
        for k, v in params.items():
            print(f"  {k:<18}: {v}")
        print()

    # YAML
    if save_path:
        with open(save_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
