import os

import torch
import wandb

from editable_stain_xaicyclegan2.model.training_controller import TrainingController
from editable_stain_xaicyclegan2.setup.settings_module import Settings
from editable_stain_xaicyclegan2.setup.wandb_module import WandbModule


def save_model(epoch, model_dir, training_controller, wandb_module, settings, prefix="", suffix=""):
    torch.save({
        'epoch': epoch,
        'wandb_step': wandb_module.step,
        'generator_he_to_p63_state_dict': training_controller.generator_he_to_p63.state_dict(),
        'generator_p63_to_he_state_dict': training_controller.generator_p63_to_he.state_dict(),
        'discriminator_he_state_dict': training_controller.discriminator_he.state_dict(),
        'discriminator_p63_state_dict': training_controller.discriminator_p63.state_dict(),
        'discriminator_he_mask_state_dict': training_controller.discriminator_he_mask.state_dict(),
        'discriminator_p63_mask_state_dict': training_controller.discriminator_p63_mask.state_dict(),
        'generator_optimizer_state_dict': training_controller.generator_optimizer.state_dict(),
        'discriminator_he_optimizer_state_dict': training_controller.discriminator_he_optimizer.state_dict(),
        'discriminator_p63_optimizer_state_dict': training_controller.discriminator_p63_optimizer.state_dict(),
        'generator_loss': training_controller.latest_generator_loss,
        'discriminator_he_loss': training_controller.latest_discriminator_he_loss,
        'discriminator_p63_loss': training_controller.latest_discriminator_p63_loss,
        'settings': settings
    }, f=os.path.join(model_dir, f'{prefix}model_checkpoint{suffix}.pth'))


def main():
    # settings = Settings('settings_test.cfg')
    settings = Settings('settings.cfg')
    wandb_module = WandbModule(settings)
    training_controller = TrainingController(settings, wandb_module)
    step_max = min(len(training_controller.train_he), len(training_controller.train_p63))
    # Directories for loading data and saving results
    data_dir = settings.data_root
    model_dir = settings.model_root
    log_dir = settings.log_dir
    # Create directories if they don't exist
    os.mkdir(log_dir) if not os.path.exists(log_dir) else None
    os.mkdir(model_dir) if not os.path.exists(model_dir) else None
    model_dir = os.path.join(model_dir, f'{settings.name}')
    os.mkdir(model_dir) if not os.path.exists(model_dir) else None
    model_file = os.path.join(model_dir, f'model_checkpoint.pth')
    # Check if model dir exists
    if os.path.exists(model_dir):
        print("Model directory: ", model_dir)
        print("Model checkpoint file: ", model_file)
    else:
        exit(1)
    model_step = 0

    for epoch in range(settings.epochs):
        # Iterate over the dataset
        for step, (real_he, real_p63) in enumerate(zip(training_controller.train_he, training_controller.train_p63)):

            # Train the model one step
            training_controller.training_step(real_he, real_p63)

            if step % settings.log_frequency == 0:  # Log every n steps
                wandb_module.log(epoch)
                wandb_module.log_image(*training_controller.get_image_pairs())
                wandb_module.step += 1

                print(f'Epoch: {epoch + 1}/{settings.epochs}\n'
                      f'Step: {step}/{step_max}\n'
                      f'Generator Loss: {training_controller.latest_generator_loss}\n'
                      f'Discriminator H&E Loss: {training_controller.latest_discriminator_he_loss}\n'
                      f'Discriminator P63 Loss: {training_controller.latest_discriminator_p63_loss}\n'
                      f'Cycle Loss: {training_controller.latest_cycle_loss}\n'
                      f'Identity Loss: {training_controller.latest_identity_loss}\n'
                      f'Context Loss {training_controller.latest_context_loss}\n'
                      f'Cycle Context Loss: {training_controller.latest_cycle_context_loss}\n'
                      )

                # Save model checkpoint
                if wandb_module.step % settings.checkpoint_frequency_steps == 0:
                    save_model(epoch, model_dir, training_controller, wandb_module, settings, suffix=f"_{model_step}")
                    model_step += 1

    # check if real_he and real_p63 exist in memory. We can't reference them directly by variable name since they may be undefined.
    if 'real_he' not in locals() or 'real_p63' not in locals():
        exit(1)

    # Export the generator_he_to_p63 model
    save_model(0, model_dir, training_controller, wandb_module, settings, prefix="final_")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        wandb.finish(0)
    except Exception as e:
        wandb.finish(1)
        raise e
