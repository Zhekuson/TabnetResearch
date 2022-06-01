<table>
	<tr>
	    <th>Датасет</th>
	    <th>n_a</th>
        <th>n_d</th>
	    <th>n_steps</th>
	    <th>Batch size</th>
	    <th>Virtual batch size</th>
	    <th>Lambda_sparse</th>
	    <th>Gamma</th>
	    <th>Momentum</th>
	</tr>
	<tr>
	    <th>Forest cover</th>
	    <th>16</th>
        <th>16</th>
	    <th>5</th>
	    <th>2048</th>
	    <th>512</th>
	    <th>1e-4</th>
	    <th>1.5</th>
	    <th>0.7</th>
	</tr>
    <tr>
	    <th>Poker hand</th>
	    <th>16</th>
        <th>16</th>
	    <th>4</th>
	    <th>4096</th>
	    <th>1024</th>
	    <th>1e-6</th>
	    <th>1.5</th>
	    <th>0.95</th>
	</tr>
    <tr>
	    <th>Sarcos</th>
	    <th>8</th>
        <th>8</th>
	    <th>3</th>
	    <th>4096</th>
	    <th>512</th>
	    <th>1e-4</th>
	    <th>1.2</th>
	    <th>0.9</th>
	</tr>
    <tr>
	    <th>Airfoil</th>
	    <th>4</th>
        <th>4</th>
	    <th>5</th>
	    <th>128</th>
	    <th>32</th>
	    <th>1e-3</th>
	    <th>1.2</th>
	    <th>0.8</th>
	</tr>    
    <tr>
	    <th>Stars</th>
	    <th>8</th>
        <th>8</th>
	    <th>3</th>
	    <th>64</th>
	    <th>32</th>
	    <th>1e-4</th>
	    <th>1.2</th>
	    <th>0.6</th>
	</tr>
    <tr>
	    <th>Syn1-Syn6</th>
	    <th colspan="8">Совпадают с оригинальной статьей TabNet</th>
	</tr>
</table>

**Для всех pretrained моделей** - _entmax_, _ratio=0.8_