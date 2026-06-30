"""Training loop: AdamW + CosineAnnealingLR + grad clip + early stopping."""
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit(model, train_loader, val_loader, train_step, val_step, device,
        epochs=80, lr=3e-4, weight_decay=5e-4, patience=15, verbose=True):
    """train_step / val_step: (model, batch, device) -> loss tensor.
    Restores best-val weights before returning the best val loss."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-6)
    best, best_state, wait = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            loss = train_step(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        vtot, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                vtot += val_step(model, batch, device).item()
                n += 1
        vloss = vtot / max(n, 1)

        if vloss < best - 1e-6:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"    early stop @ epoch {ep + 1} (best val {best:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best
