r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 128
    hypers["seq_len"] = 64
    hypers["h_dim"] = 512
    hypers["n_layers"] = 2
    hypers["dropout"] = 0.11
    hypers["learn_rate"] = 0.0005
    hypers["lr_sched_factor"] = 0.08
    hypers["lr_sched_patience"] = 1
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I.\n Scene I.\n"
    temperature= 0.51
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

We split the corpus into sequences instead of training on the whole text for several reasons. This approach helps maintain contextual relationships between nearby sentences and avoids learning irrelevant patterns from distant parts of the text. It also conserves memory, speeds up training, and reduces the risk of vanishing or exploding gradients. Overall, splitting the corpus into sequences improves learning efficiency and the model's ability to capture localized context.
"""

part1_q2 = r"""
**Your answer:**

The generated text can show memory longer than the sequence length because the model uses hidden states to remember information from previous batches. These hidden states keep track of earlier context and carry it forward, allowing the model to generate text that makes sense over longer spans. By passing these hidden states through successive batches, the model can maintain coherence and context beyond the length of any single sequence.
"""

part1_q3 = r"""
**Your answer:**

We don’t shuffle the order of batches because keeping the sequence intact is crucial for preserving the meaning and context of the text. Language relies on the order of sentences to convey meaning, and shuffling batches would disrupt this flow, leading to a loss of coherence. By maintaining the order, we ensure that hidden states effectively carry relevant information from one batch to the next, helping the model generate text that makes sense and follows the intended structure.
"""

part1_q4 = r"""
**Your answer:**

1. We lower the temperature during sampling to manage the balance between creativity and coherence in the generated text. A lower temperature makes the probability distribution more peaked, meaning the model favors high-probability words more strongly. This results in more predictable and coherent text because the model's choices are more focused and aligned with its confident predictions. Lowering the temperature helps to avoid generating nonsensical or irrelevant text that might occur if the model takes too many risks.

2.With a very high temperature, the probability distribution becomes nearly uniform, meaning that all words or characters have similar chances of being selected. This randomness encourages exploration and diversity in the text, but it often results in less coherent and meaningful output. High temperatures reduce the influence of word probabilities, leading to more random and unpredictable text that may lack structure and context.

3.When the temperature is very low, the probability distribution becomes highly peaked or almost one-hot encoded. This means the model predominantly selects the word with the highest probability, making its output very deterministic. While this can produce highly coherent text, it also reduces variety and creativity. Extremely low temperatures can lead to repetitive text and may cause the model to generate similar or looping phrases, as it relies heavily on a narrow set of choices.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.5, 0.999)
    # ========================
    return hypers

def part3_gan_hyperparams():
    hypers = dict(
        batch_size=8,
        z_dim=16,
        data_label=1,
        label_noise=0.08,
        discriminator_optimizer=dict(
            type="Adam", 
            lr=0.001,
            betas=(0.5, 0.999),
            
        ),
        generator_optimizer=dict(
            type="Adam",  
            lr=0.0005,
            betas=(0.5, 0.999),
            
        ),
    )
    return hypers

part2_q1 = r"""

In GAN training, the need to maintain or discard gradients depends on whether the discriminator or generator is being trained.

When training the \textbf{discriminator}, the goal is to improve its ability to distinguish between real and fake images. Here, we use the generator to produce fake images but \textbf{do not} calculate or propagate gradients back through the generator. This is because updating the generator during discriminator training could degrade the generator’s performance, making it worse at generating convincing images. The discriminator’s loss is minimized, which would inadvertently "hurt" the generator if its gradients were kept. Therefore, we discard the generator’s gradients to avoid affecting its parameters.

Conversely, when training the \textbf{generator}, maintaining gradients is crucial. The generator’s objective is to improve its ability to produce realistic images that can fool the discriminator. To achieve this, we calculate and propagate gradients back through the generator, allowing it to update its parameters and enhance its image generation capabilities.


"""

part2_q2 = r"""
1. Stopping GAN training based solely on the generator's loss dropping below a certain threshold is not a good idea because the loss alone does not give a full picture of the GAN's performance. The generator's loss measures how well it fools the discriminator, but it doesn't directly reflect the quality of the generated images. As training progresses, the generator might find ways to exploit the discriminator's weaknesses, leading to a low loss without necessarily producing realistic images. Additionally, as the discriminator improves, the generator's task becomes harder, and its loss might fluctuate. Effective GAN training requires monitoring both the generator and discriminator losses, as well as the quality of the generated outputs, rather than relying solely on a numerical threshold.

2. If the discriminator's loss remains constant while the generator's loss decreases, it may indicate that the generator has learned to exploit the discriminator in a way that consistently produces outputs the discriminator cannot distinguish from real images. This situation can lead to the discriminator no longer learning effectively because it isn't receiving informative gradients that would allow it to improve its ability to classify real and fake images. This is a sign of mode collapse, where the generator produces a limited variety of outputs that successfully trick the discriminator, but the overall diversity and quality of generated images may suffer. The generator's loss decreases because it's getting better at fooling the discriminator, but the discriminator isn't keeping up, which can stall the GAN's progress and lead to suboptimal outcomes.

"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


part4_q3= r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""


# ==============
