import torch

def supervised_train_step(e, i, train_iterator, val_loader, cinn, train_params, optimizer, scheduler, ndim_total, device, loss_log, **config):
    im, cond = next(train_iterator)

    im = im.to(device)
    cond = cond.to(device)

    out, log_j = cinn(im, c=cond)

    alt_nll = torch.mean(out ** 2 / 2) - torch.mean(log_j) / ndim_total

    nll = 0.5 * torch.sum(out ** 2, dim=1) - log_j
    nll = torch.mean(nll) / ndim_total

    if config.get("grad_loss"):
        from imageflow.losses import Grad
        penalty = config.get('grad_loss')['penalty']
        grad_lambda = config.get("grad_loss")["multiplier"]
        smoother = Grad(penalty=penalty, mult=grad_lambda)
        smooth_reg = smoother.loss(out)
        nll += smooth_reg

    alt_nll.backward()

    # print("{}\t{}\t{}".format(e, i, alt_nll.item()))

    torch.nn.utils.clip_grad_norm_(train_params, 100.)

    optimizer.step()
    optimizer.zero_grad()
    # scheduler.step()

    if i % 20 == 0:
        with torch.no_grad():
            val_x, val_c = next(iter(val_loader))
            val_x, val_c = val_x.to(device), val_c.to(device)

            v_out, v_log_j = cinn(val_x, c=val_c)
            v_nll = torch.mean(v_out ** 2) / 2 - torch.mean(v_log_j) / ndim_total
            loss_log['nll'].append(alt_nll.item())
            loss_log['val_nll'].append(v_nll.item())
            loss_log['epoch'].append(e)
            loss_log["batch"].append(i)
            loss_log['lr'].append(scheduler.get_last_lr())
            print("{}\t{}\t{}\t{}\t{}\t{}".format(e, i, alt_nll.item(), nll.item(), v_nll.item(),
                                                  scheduler.get_last_lr()))
            # agg_nll.append(alt_nll)


def unsupervised_training_step(e, i, unsup_iterator, unsup_val_loader, cinn, train_params, optimizer, scheduler, ndim_total, device, output_log, **config):
    # unsupervised training step
    unsup_batch = next(unsup_iterator)
    batch_size = unsup_batch.shape[0]
    unsup_v_field, unsup_cond = unsup_batch
    unsup_cond = unsup_cond.to(device)
    # source = cond[:, :1, ...].to(device)
    target = unsup_cond[:, 1:, ...].to(device)
    z = torch.randn(batch_size, ndim_total).to(device)
    target_pred, v_field_pred, log_jac = cinn.reverse_sample(z, unsup_cond)
    rec_term = torch.mean(torch.mean(torch.square(target_pred - target), dim=(1, 2, 3)))
    loss = rec_term

    if config.get("grad_loss"):
        from imageflow.losses import Grad
        # print("check")
        penalty = config.get('grad_loss')['penalty']
        grad_lambda = config.get("grad_loss")["multiplier"]
        smoother = Grad(penalty=penalty, mult=grad_lambda)
        smooth_reg = smoother.loss(v_field_pred)
        loss += smooth_reg
    if config.get("curl"):
        from imageflow.losses import Rotation
        multiplier = config.get("curl")["multiplier"]
        curl_loss = Rotation(multiplier)
        smooth_reg = curl_loss.loss(v_field_pred)
        loss += smooth_reg

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(train_params, 10.)
    optimizer.step()

    if i % 20 == 0:

        if config.get("data") == "Birl":
            val_cond = next(iter(unsup_val_loader))
        else:
            _, val_cond = next(iter(unsup_val_loader))
        val_batch_size = val_cond.shape[0]
        val_cond = val_cond.to(device)
        val_target = val_cond[:, 1:, ...].to(device)
        z = torch.randn(val_batch_size, ndim_total).to(device)
        val_target_pred, val_v_field_pred, val_log_jac = cinn.reverse_sample(z, val_cond)
        val_rec_term = torch.mean(torch.mean(torch.square(val_target_pred - val_target), dim=(1, 2, 3)))
        val_loss = round(val_rec_term.item(), 2)
        loss_out = round(loss.item(), 2)
        if loss_out == 0.00 or val_loss == 0.00:
            loss_out = "{:.1e}".format(loss.item())
            val_loss = "{:.1e}".format(val_rec_term.item())
        output = "unsupervised: {}\t\t{}\t\t{}\t\t{}".format(e, i, loss_out,
                                                             val_loss)  # , rec_out, prior_out, jac_out, f)
        if config.get("grad_loss") or config.get('curl'):
            output += f"\t\t{smooth_reg}"
        output_log += (output + "\n")
        output_log += str(scheduler.get_last_lr())
        print(output)
        if config.get("show_running"):
            from imageflow.visualize import plot_running_fields
            from imageflow.visualize import plot_mff_
            # streamplot_from_batch(val_v_field_pred.cpu().detach().numpy())
            plot_running_fields(val_v_field_pred.cpu().detach().numpy(), idx=0)
            plot_mff_(val_cond.cpu().detach().numpy(), val_target_pred.cpu().detach().numpy(), idx=0)
