import torch
from trl.trainer.sft_trainer import SFTTrainer


class PrunedLoRATrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.lora_lambda1: float = kwargs.pop("lora_lambda1", 0.01)
        self.lora_lambda2: float = kwargs.pop("lora_lambda2", 0.01)
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        res = super().compute_loss(
            model=model,
            inputs=inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

        if return_outputs:
            assert isinstance(res, tuple)
            loss = res[0]
        else:
            loss = res

        reg_loss = torch.zeros_like(loss)
        l1 = self.lora_lambda1
        l2 = self.lora_lambda2

        if l1 > 0 or l2 > 0:
            for name, param in model.named_parameters():
                if "lora_A" in name and l2 > 0:
                    # Y is (r, d_in), column-wise norm corresponds to input features
                    reg_loss += l2 * torch.norm(param, p=2, dim=0).sum()
                elif "lora_B" in name and l1 > 0:
                    # B is (d_out, r), row-wise norm corresponds to output features
                    reg_loss += l1 * torch.norm(param, p=2, dim=1).sum()

        loss += reg_loss
        if return_outputs:
            res[0] = loss
            return res
        return loss
