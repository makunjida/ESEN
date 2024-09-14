"""
Base model for all matching model所有匹配模型的基础模型
"""

from torch import nn
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine import hyper_spaces
import typing
from matchzoo.utils import parse
import numpy as np
import torch
import torch_utils as my_utils


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def get_default_params(self,
                           with_embedding = False,
                           with_multi_layer_perceptron = False):
        """
        Model default parameters.模型的默认参数

        The common usage is to instantiate :class:`matchzoo.engine.ModelParams`常见的用法是首先实例化 matchzoo.engine.ModelParams 对象，然后再设置模型特定的参数。
            first, then set the model specific parametrs.

        Examples:
            >>> class MyModel(BaseModel):
            ...     def build(self):
            ...         print(self._params['num_eggs'], 'eggs')
            ...         print('and', self._params['ham_type'])
            ...
            ...
            ...     def get_default_params(self):
            ...         params = ParamTable()
            ...         params.add(Param('num_eggs', 512))
            ...         params.add(Param('ham_type', 'Parma Ham'))
            ...         return params
            >>> my_model = MyModel()
            >>> my_model.build()
            512 eggs
            and Parma Ham

        Notice that all parameters must be serialisable for the entire model
        to be serialisable. Therefore, it's strongly recommended to use python
        native data types to store parameters.#请注意,为了使整个模型都可以被序列化,所有的参数都必须是可序列化的。因此,强烈建议使用Python的本机数据类型来存储参数。

        :return: model parameters

        """
        params = ParamTable()
        params.add(Param(
            name = 'model_class', value = self.__class__.__name__,
            desc = "Model class. Used internally for save/load. "
                   "Changing this may cause unexpected behaviors."
        ))
        params.add(Param(
            name = 'input_shapes',
            desc = "Dependent on the model and data. Should be set manually."
        ))
        params.add(Param(
            name = 'task',
            desc = "Decides model output shape, loss, and metrics."
        ))
        params.add(Param(
            name = 'optimizer', value = 'adam',
        ))
        if with_embedding:
            params.add(Param(
                name = 'with_embedding', value = True,
                desc = "A flag used help `auto` module. Shouldn't be changed."
            ))
            params.add(Param(
                name = 'embedding_input_dim',
                desc = 'Usually equals vocab size + 1. Should be set manually.'
            ))
            params.add(Param(
                name = 'embedding_output_dim',
                desc = 'Should be set manually.'
            ))
            params.add(Param(
                name = 'embedding_trainable', value = True,
                desc = '`True` to enable embedding layer training, '
                       '`False` to freeze embedding parameters.'
            ))
        if with_multi_layer_perceptron:
            params.add(Param(
                name = 'with_multi_layer_perceptron', value = True,
                desc = "A flag of whether a multiple layer perceptron is used. "
                       "Shouldn't be changed."
            ))
            params.add(Param(
                name = 'mlp_num_units', value = 128,
                desc = "Number of units in first `mlp_num_layers` layers.",
                hyper_space = hyper_spaces.quniform(8, 256, 8)
            ))
            params.add(Param(
                name = 'mlp_num_layers', value = 3,
                desc = "Number of layers of the multiple layer percetron.",
                hyper_space = hyper_spaces.quniform(1, 6)
            ))
            params.add(Param(
                name = 'mlp_num_fan_out', value = 64,
                desc = "Number of units of the layer that connects the multiple "
                       "layer percetron and the output.",
                hyper_space = hyper_spaces.quniform(4, 128, 4)
            ))
            params.add(Param(
                name = 'mlp_activation_func', value = 'relu',
                desc = 'Activation function used in the multiple '
                       'layer perceptron.'
            ))
        return params

    def _make_perceptron_layer(#用于创建感知机层，通过特征提取和非线性建模能力，使得神经网络能够适应不同的数据模式和任务，并实现高效的数据表示和预测能力。
        self,
        in_features: int = 0,
        out_features: int = 0,
        activation: nn.Module = nn.ReLU
    ) -> nn.Module:
        """:return: a perceptron layer."""
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            activation
        )

    def _make_output_layer(#根据需要创建模型的输出层，可以选择是否添加激活函数。
        self,
        in_features: int = 0,
        activation: typing.Union[str, nn.Module] = None
    ) -> nn.Module:
        """:return: a correctly shaped torch module for model output."""
        if activation:
            return nn.Sequential(
                nn.Linear(in_features, 1),
                parse.parse_activation(activation)
            )
        else:
            return nn.Linear(in_features, 1)

    def _make_default_embedding_layer(#根据提供的参数创建默认的嵌入层，
        self, _params) -> nn.Module:
        """:return: an embedding module."""
        if isinstance(_params['embedding'], np.ndarray):
            _params['embedding_input_dim'] = (#计算嵌入矩阵的输入维度 _params['embedding_input_dim']，这通常是嵌入矩阵的行数（表示词汇表中的单词数）
                _params['embedding'].shape[0]
            )
            _params['embedding_output_dim'] = (#计算嵌入矩阵的输出维度 _params['embedding_output_dim']，这通常是嵌入向量的维度。
                _params['embedding'].shape[1]
            )
            return nn.Embedding.from_pretrained(
                embeddings=torch.Tensor(_params['embedding']),
                freeze=_params['embedding_freeze']
            )
        else:
            return nn.Embedding(
                num_embeddings=_params['embedding_input_dim'],
                embedding_dim=_params['embedding_output_dim']
            )

    def _make_default_char_embedding_layer(#根据提供的参数创建默认的字符嵌入层，这个嵌入层模块可以将输入的单词索引映射为相应的嵌入向量
        self, _params) -> nn.Module:
        """:return: an embedding module."""
        if isinstance(_params['char_embedding'], np.ndarray):
            _params['char_embedding_input_dim'] = (
                _params['char_embedding'].shape[0]
            )
            _params['char_embedding_output_dim'] = (
                _params['char_embedding'].shape[1]
            )
            return nn.Embedding.from_pretrained(
                embeddings=torch.Tensor(_params['char_embedding']),
                freeze=_params['char_embedding_freeze']
            )
        else:
            return nn.Embedding(
                num_embeddings=_params['char_embedding_input_dim'],
                embedding_dim=_params['char_embedding_output_dim']
            )

    def _make_entity_embedding_layer(#创建实体嵌入层
        self, matrix: np.ndarray, freeze: bool) -> nn.Module:
        """:return: an embedding module."""
        return nn.Embedding.from_pretrained(
            embeddings = torch.Tensor(matrix), freeze = freeze)

    def predict(self, query: np.ndarray, doc: np.ndarray, verbose: bool = False, **kargs) -> np.ndarray:#用于进行预测
        self.train(False)  # very important, to disable dropout
        if verbose:
            print("query: ", query)
            print("doc: ", doc)
            print("================ end of query doc =================")
        out = self(query, doc, verbose, **kargs)
        return my_utils.cpu(out).detach().numpy().flatten()

    def forward(self, *input):
        pass


if __name__ == '__main__':
    print("here")