import torch

from vae_project.losses import kl_divergence_standard_normal, reconstruction_bce_with_logits, vae_loss
from vae_project.models import MLPVAE


def test_mlp_vae_forward_shapes():
    model = MLPVAE(input_shape=(1, 28, 28), hidden_dims=[32, 16], latent_dim=4)
    x = torch.rand(5, 1, 28, 28)

    output = model(x)

    assert output["recon_logits"].shape == x.shape
    assert output["mu"].shape == (5, 4)
    assert output["logvar"].shape == (5, 4)
    assert output["z"].shape == (5, 4)


def test_reparameterize_eval_uses_mu():
    model = MLPVAE(input_shape=(1, 28, 28), hidden_dims=[16], latent_dim=3)
    model.eval()
    mu = torch.ones(2, 3)
    logvar = torch.zeros(2, 3)

    z = model.reparameterize(mu, logvar)

    assert torch.equal(z, mu)


def test_reparameterize_can_sample_in_eval_mode():
    model = MLPVAE(input_shape=(1, 28, 28), hidden_dims=[16], latent_dim=3)
    model.eval()
    mu = torch.ones(2, 3)
    logvar = torch.zeros(2, 3)
    torch.manual_seed(17)
    expected = mu + torch.randn_like(mu)

    torch.manual_seed(17)
    z = model.reparameterize(mu, logvar, sample=True)

    assert torch.equal(z, expected)


def test_reparameterize_can_use_mu_in_train_mode():
    model = MLPVAE(input_shape=(1, 28, 28), hidden_dims=[16], latent_dim=3)
    model.train()
    mu = torch.ones(2, 3)
    logvar = torch.zeros(2, 3)

    z = model.reparameterize(mu, logvar, sample=False)

    assert torch.equal(z, mu)


def test_kl_zero_for_standard_normal_posterior():
    mu = torch.zeros(7, 4)
    logvar = torch.zeros(7, 4)

    kl = kl_divergence_standard_normal(mu, logvar)

    assert torch.isclose(kl, torch.tensor(0.0))


def test_reconstruction_bce_is_scalar():
    logits = torch.zeros(3, 1, 28, 28)
    target = torch.zeros(3, 1, 28, 28)

    loss = reconstruction_bce_with_logits(logits, target)

    assert loss.ndim == 0
    assert loss.item() > 0


def test_vae_loss_combines_terms_with_beta():
    logits = torch.zeros(2, 1, 28, 28)
    target = torch.rand(2, 1, 28, 28)
    mu = torch.ones(2, 4)
    logvar = torch.zeros(2, 4)

    beta0 = vae_loss(logits, target, mu, logvar, beta=0.0)
    beta1 = vae_loss(logits, target, mu, logvar, beta=1.0)

    assert torch.isclose(beta0["total"], beta0["reconstruction"])
    assert torch.isclose(beta1["total"], beta1["reconstruction"] + beta1["kl"])
    assert beta1["kl"].item() > 0
