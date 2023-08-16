import numpy as np
import torch

from disney_route_optimize.wait_predction.model.transformer.mask import (
    create_mask,
    generate_square_subsequent_mask,
)


def del_zero(tensor):
    # 1次元に変換
    tensor = tensor.view(-1)

    # 末尾の連続した0を削除
    nonzero_indices = (tensor != 0).nonzero()
    if len(nonzero_indices) > 0:
        last_nonzero_index = nonzero_indices[-1].item()
        tensor = tensor[: last_nonzero_index + 1]

    # 元のtorch.Sizeに戻す
    tensor = tensor.view([1, tensor.shape[0], 1])
    return tensor


def train(model, data_loader, optimizer, criterion, device: str):
    model.train()
    total_loss = []
    for src, tgt in data_loader:
        src = src.float().to(device)
        tgt = tgt.float().to(device)
        src = del_zero(src)
        tgt = tgt[:, : src.shape[1], :]

        input_tgt = torch.cat((src[:, -1:, :], tgt[:, :-1, :]), dim=1)

        mask_src, mask_tgt = create_mask(src, input_tgt, device)

        output = model(src=src, tgt=input_tgt, mask_src=mask_src, mask_tgt=mask_tgt)

        optimizer.zero_grad()

        loss = criterion(output, tgt)
        loss.backward()
        total_loss.append(loss.cpu().detach())
        optimizer.step()

    return np.average(total_loss)


def evaluate(model, data_loader, criterion, device: str):
    model.eval()
    total_loss = []
    for src, tgt in data_loader:
        src = src.float().to(device)
        tgt = tgt.float().to(device)
        src = del_zero(src)
        tgt = tgt[:, : src.shape[1], :]

        seq_len_src = src.shape[1]
        mask_src = (torch.zeros(seq_len_src, seq_len_src)).type(torch.bool)
        mask_src = mask_src.float().to(device)

        memory = model.encode(src, mask_src)
        outputs = src[:, -1:, :]
        seq_len_tgt = tgt.shape[1]

        for i in range(seq_len_tgt - 1):
            mask_tgt = (generate_square_subsequent_mask(outputs.size(1))).to(device)

            output = model.decode(outputs, memory, mask_tgt)
            output = model.output(output)

            outputs = torch.cat([outputs, output[:, -1:, :]], dim=1)

        loss = criterion(outputs, tgt)
        total_loss.append(loss.cpu().detach())

    return np.average(total_loss)


def test(model, data_loader, criterion, device: str):
    model.eval()
    total_loss = []
    list_output = []
    list_tgt = []
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.float().to(device)
            tgt = tgt.float().to(device)
            src = del_zero(src)
            tgt = tgt[:, : src.shape[1], :]

            seq_len_src = src.shape[1]
            mask_src = (torch.zeros(seq_len_src, seq_len_src)).type(torch.bool)
            mask_src = mask_src.float().to(device)

            memory = model.encode(src, mask_src)
            outputs = src[:, -1:, :]
            seq_len_tgt = tgt.shape[1]

            for i in range(seq_len_tgt - 1):
                mask_tgt = (generate_square_subsequent_mask(outputs.size(1))).to(device)

                output = model.decode(outputs, memory, mask_tgt)
                output = model.output(output)

                outputs = torch.cat([outputs, output[:, -1:, :]], dim=1)

            loss = criterion(outputs, tgt)
            total_loss.append(loss.cpu().detach())
            list_output.append(list(outputs.view(-1).to("cpu").detach().numpy()))
            list_tgt.append(list(tgt.view(-1).to("cpu").detach().numpy()))

    return np.average(total_loss), list_output, list_tgt
