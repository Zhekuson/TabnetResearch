from sdv.tabular import CTGAN
from sdv.evaluation import evaluate
from sdv.tabular import TVAE
from sdv.tabular.base import DISABLE_TMP_FILE


class CommonObjective:

    def __init__(self):
        pass

    def __call__(self, trial):
        pass


class CTGANObjective(CommonObjective):
    discriminator_dims_available = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
    generator_dims_available = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]

    def __init__(self, data):
        super().__init__()
        self.best_model = None
        self._models = []
        self.data = data

    def __call__(self, trial):
        embedding_dim = trial.suggest_int('embedding_dim', 64, 256)

        generator_dims_index = trial.suggest_categorical("generator_dims_index", [0, 1, 2, 3, 4, 5])
        generator_dims = CTGANObjective.generator_dims_available[generator_dims_index]
        generator_lr = trial.suggest_float('generator_lr', 1e-6, 1e-2)
        generator_decay = trial.suggest_float('generator_decay', 1e-7, 1e-3)

        discriminator_dims_index = trial.suggest_categorical("discriminator_dims_index", [0, 1, 2, 3, 4, 5])
        discriminator_dims = CTGANObjective.discriminator_dims_available[discriminator_dims_index]
        discriminator_lr = trial.suggest_float('discriminator_lr', 1e-6, 1e-2)
        discriminator_decay = trial.suggest_float('discriminator_decay', 1e-7, 1e-3)

        batch_size = trial.suggest_int('batch_size', 10, 400, step=10)  # always multiple of 10
        epochs = trial.suggest_int('epochs', 20, 300)
        model = CTGAN(cuda=True, embedding_dim=embedding_dim, batch_size=batch_size,
                      discriminator_dim=discriminator_dims, epochs=epochs,
                      generator_lr=generator_lr, discriminator_lr=discriminator_lr,
                      generator_dim=generator_dims, generator_decay=generator_decay,
                      discriminator_decay=discriminator_decay)
        self._models.append((model, trial))
        model.fit(self.data)
        gen_data = model.sample(len(self.data), output_file_path=DISABLE_TMP_FILE)
        return evaluate(gen_data, self.data)

    @staticmethod
    def params_to_kwargs(params):
        return {
            'embedding_dim': params['embedding_dim'],
            'batch_size': params['batch_size'],
            'discriminator_dim': CTGANObjective.discriminator_dims_available[params['discriminator_dims_index']],
            'epochs': params['epochs'],
            'generator_lr': params['generator_lr'],
            'discriminator_lr': params['discriminator_lr'],
            'generator_dim': CTGANObjective.generator_dims_available[params['generator_dims_index']],
            'generator_decay': params['generator_decay'],
            'discriminator_decay': params['discriminator_decay']
        }


class TVAEObjective(CommonObjective):
    compress_dims_available = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
    decompress_dims_available = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]

    def __init__(self, data):
        super().__init__()
        self.best_model = None
        self._models = []
        self.data = data

    def __call__(self, trial):
        embedding_dim = trial.suggest_int('embedding_dim', 64, 256)
        batch_size = trial.suggest_int('batch_size', 10, 400, step=10)  # always multiple of 10
        epochs = trial.suggest_int('epochs', 20, 300)

        compress_dims_index = trial.suggest_categorical("compress_dims_index", [0, 1, 2, 3, 4, 5])
        compress_dims = TVAEObjective.compress_dims_available[compress_dims_index]

        decompress_dims_index = trial.suggest_categorical("decompress_dims_index", [0, 1, 2, 3, 4, 5])
        decompress_dims = TVAEObjective.decompress_dims_available[decompress_dims_index]

        l2scale = trial.suggest_float('l2scale', 1e-7, 1e-3)
        loss_factor = trial.suggest_float('loss_factor', 1e-5, 1e2)

        model = TVAE(cuda=True, embedding_dim=embedding_dim, batch_size=batch_size,
                     epochs=epochs, compress_dims=compress_dims, decompress_dims=decompress_dims,
                     l2scale=l2scale, loss_factor=loss_factor)
        self._models.append((model, trial))
        model.fit(self.data)
        gen_data = model.sample(len(self.data), output_file_path=DISABLE_TMP_FILE)
        return evaluate(gen_data, self.data)

    @staticmethod
    def params_to_kwargs(params):
        return {
            'embedding_dim': params['embedding_dim'],
            'batch_size': params['batch_size'],
            'epochs': params['epochs'],
            'l2scale': params['l2scale'],
            'loss_factor': params['loss_factor'],
            'compress_dims': TVAEObjective.compress_dims_available[params['compress_dims_index']],
            'decompress_dims': TVAEObjective.decompress_dims_available[params['decompress_dims_index']]
        }
