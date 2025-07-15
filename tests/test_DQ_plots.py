from src.mspc_pca.mspc import plot_DQ, plot_DQ_tt

import numpy as np
import matplotlib.pyplot as plt
import pytest

@pytest.fixture
def sample_data_single():
    """Provides sample data for tests of plot_DQ."""
    D = np.random.rand(20) * 10
    Q = np.random.rand(20) * 5
    threshold_D = 7.0
    threshold_Q = 3.0
    return D, Q, threshold_D, threshold_Q

@pytest.fixture
def sample_data():
    """Provides sample data for tests of plot_DQ_tt."""
    D_train = np.random.rand(20) * 10
    Q_train = np.random.rand(20) * 5
    D_test = np.random.rand(10) * 10
    Q_test = np.random.rand(10) * 5
    threshold_D = 7.0
    threshold_Q = 3.0
    return D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q

# @pytest.fixture(autouse=True)
# def close_plots_after_test():
#     """Ensures all matplotlib figures are closed after each test."""
#     yield
#     plt.close('all')

class TestPlotDQ:
    def test_basic_plot(self, sample_data_single):
        """Test basic plotting of D and Q."""
        D, Q, threshold_D, threshold_Q = sample_data_single
        
        fig, axes = plot_DQ(D, Q, threshold_D, threshold_Q)
        plt.tight_layout()
        plt.show()
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 2
        assert isinstance(axes[0], plt.Axes)
        assert isinstance(axes[1], plt.Axes)
        assert axes[0].get_title() == "D-statistic"
        assert axes[1].get_title() == "Q-statistic"
        assert len(axes[0].patches) == len(D) 
        assert len(axes[1].patches) == len(Q)
        assert len(axes[0].lines) >= 1 # At least one threshold line
        assert len(axes[1].lines) >= 1 # At least one threshold line

    def test_with_event_index(self, sample_data_single):
        """Test highlighting specific event indices."""
        D, Q, threshold_D, threshold_Q = sample_data_single
        event_indices = [5, 19]
        
        fig, axes = plot_DQ(D, Q, threshold_D, threshold_Q, event_index=event_indices)
        plt.tight_layout()
        plt.show()
        
        assert len(axes[0].patches) == len(D)
        assert len(axes[1].patches) == len(Q)

    def test_logscale(self, sample_data_single):
        """Test plotting with logarithmic scale."""
        D, Q, threshold_D, threshold_Q = sample_data_single
        
        fig, axes = plot_DQ(D, Q, threshold_D, threshold_Q, logscale=True)
        plt.tight_layout()
        plt.show()
        
        assert axes[0].get_yscale() == 'log'
        assert axes[1].get_yscale() == 'log'

    def test_scalar_thresholds(self, sample_data_single):
        """Test with scalar thresholds."""
        D, Q, threshold_D, threshold_Q = sample_data_single
        
        fig, axes = plot_DQ(D, Q, threshold_D, threshold_Q)
        plt.tight_layout()
        plt.show()
        
        # Check for D threshold line
        d_threshold_lines = [line for line in axes[0].lines if 'D threshold' in line.get_label()]
        assert len(d_threshold_lines) == 1
        assert d_threshold_lines[0].get_ydata()[0] == threshold_D

        # Check for Q threshold line
        q_threshold_lines = [line for line in axes[1].lines if 'Q threshold' in line.get_label()]
        assert len(q_threshold_lines) == 1
        assert q_threshold_lines[0].get_ydata()[0] == threshold_Q

    def test_list_thresholds(self, sample_data_single):
        """Test with list thresholds and alpha values."""
        D, Q, _, _ = sample_data_single
        threshold_D_list = [5.0, 8.0]
        threshold_Q_list = [2.0, 4.0]
        alpha_values = [0.01, 0.05]
        
        fig, axes = plot_DQ(D, Q, threshold_D_list, threshold_Q_list, alpha=alpha_values, type_q='Custom')
        plt.tight_layout()
        plt.show()
        
        # Check for D threshold lines
        d_threshold_lines = [line for line in axes[0].lines if 'D threshold' in line.get_label()]
        assert len(d_threshold_lines) == len(threshold_D_list)
        # Check that the y-values of the lines match the thresholds
        assert sorted([line.get_ydata()[0] for line in d_threshold_lines]) == sorted(threshold_D_list)
        # Check labels for alpha values
        assert any(f'$\\alpha$={alpha_values[0]}' in line.get_label() for line in d_threshold_lines)
        assert any(f'$\\alpha$={alpha_values[1]}' in line.get_label() for line in d_threshold_lines)


        # Check for Q threshold lines
        q_threshold_lines = [line for line in axes[1].lines if 'Q threshold' in line.get_label()]
        assert len(q_threshold_lines) == len(threshold_Q_list)
        assert sorted([line.get_ydata()[0] for line in q_threshold_lines]) == sorted(threshold_Q_list)
        # Check labels for alpha values and type_q
        assert any(f'$\\alpha$={alpha_values[0]}' in line.get_label() for line in q_threshold_lines)
        assert any(f'$\\alpha$={alpha_values[1]}' in line.get_label() for line in q_threshold_lines)
        assert all('Custom' in line.get_label() for line in q_threshold_lines)


    def test_with_provided_axes(self, sample_data_single):
        """Test plotting on pre-existing matplotlib Axes objects."""
        D, Q, threshold_D, threshold_Q = sample_data_single
        
        # Create a figure and axes manually
        fig_manual, (ax_d_manual, ax_q_manual) = plt.subplots(2, 1)
        
        # Pass these axes to the function
        returned_fig, returned_axes = plot_DQ(D, Q, threshold_D, threshold_Q, ax=[ax_d_manual, ax_q_manual])
        plt.tight_layout()
        plt.show()
        
        assert fig_manual == returned_fig
        assert returned_axes[0] is ax_d_manual
        assert returned_axes[1] is ax_q_manual
        
        # Check if content was added to the provided axes
        assert len(ax_d_manual.patches) > 0
        assert len(ax_q_manual.patches) > 0

    def test_invalid_ax_input(self):
        """Test that a ValueError is raised for invalid 'ax' input."""
        D = np.array([1])
        Q = np.array([1])
        threshold_D = 1
        threshold_Q = 1

        # Test with a single ax instead of a list/tuple of two
        fig_single, ax_single = plt.subplots()
        with pytest.raises(ValueError, match="If 'ax' is provided, it must be a list or tuple of two matplotlib Axes objects."):
            plot_DQ(D, Q, threshold_D, threshold_Q, ax=ax_single)
        plt.close(fig_single) 

        # Test with a list of incorrect length
        fig_bad_len, ax_bad_len = plt.subplots(3, 1)
        with pytest.raises(ValueError, match="If 'ax' is provided, it must be a list or tuple of two matplotlib Axes objects."):
            plot_DQ(D, Q, threshold_D, threshold_Q, ax=ax_bad_len[0:1]) # Pass only one ax from the array
        plt.close(fig_bad_len) 

        # Test with non-list/tuple input
        with pytest.raises(ValueError, match="If 'ax' is provided, it must be a list or tuple of two matplotlib Axes objects."):
            plot_DQ(D, Q, threshold_D, threshold_Q, ax="invalid")

    def test_with_labels(self, sample_data_single):
        """Test plotting with custom x-axis labels."""
        D, Q, threshold_D, threshold_Q = sample_data_single
        total_len = len(D)
        labels = [f'Step_{i}' for i in range(total_len)]
        
        fig, axes = plot_DQ(D, Q, threshold_D, threshold_Q, labels=labels)
        plt.tight_layout()
        plt.show()
        
        # Check if x-tick labels are set
        assert len(axes[0].get_xticklabels()) == total_len
        assert axes[0].get_xticklabels()[0].get_text() == 'Step_0'
        assert axes[0].get_xticklabels()[-1].get_text() == f'Step_{total_len - 1}'
        assert axes[1].get_xticklabels()[0].get_text() == 'Step_0'

    def test_opacity(self, sample_data_single):
        """Test applying opacity to bars."""
        D, Q, threshold_D, threshold_Q = sample_data_single
        opacity_values = np.linspace(0.1, 1.0, len(D))
        
        fig, axes = plot_DQ(D, Q, threshold_D, threshold_Q, opacity=opacity_values)
        plt.tight_layout()
        plt.show()
        
        assert len(axes[0].patches) == len(D)
        assert len(axes[1].patches) == len(Q)

        # Check if opacity was set
        first_patch_color_d = axes[0].patches[0].get_facecolor()
        assert len(first_patch_color_d) == 4 # RGBA tuple


class TestPlotDQtt:
    def test_basic_plot_train_test(self, sample_data):
        """Test basic plotting with both train and test data."""
        D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q = sample_data
        
        fig, axes = plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q)
        plt.tight_layout()
        plt.show()
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 2
        assert isinstance(axes[0], plt.Axes)
        assert isinstance(axes[1], plt.Axes)
        assert axes[0].get_title() == "D-statistic"
        assert axes[1].get_title() == "Q-statistic"
        assert len(axes[0].patches) == 30 # 20 train and 10 test
        assert len(axes[1].patches) == 30 # 20 train and 10 test
        assert len(axes[0].lines) >= 1 # At least one threshold line
        assert len(axes[1].lines) >= 1 # At least one threshold line


    def test_plot_only_test(self, sample_data):
        """Test plotting with only test data."""
        D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q = sample_data
        
        fig, axes = plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q, plot_train=False)
        plt.tight_layout()
        plt.show()
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 2
        assert len(axes[0].patches) == 10 # Only 10 bars (test)
        assert len(axes[1].patches) == 10 # Only 10 bars (test)
        # Check that the train/test separator line is NOT present when plot_train is False
        assert not any(line.get_label() == 'Train/Test Split' for line in axes[0].lines)
        assert not any(line.get_label() == 'Train/Test Split' for line in axes[1].lines)


    def test_with_labels(self, sample_data):
        """Test plotting with custom x-axis labels."""
        D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q = sample_data
        total_len = len(D_train) + len(D_test)
        labels = [f'Label_{i}' for i in range(total_len)]
        
        fig, axes = plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q, labels=labels)
        axes[0].set_title('Overwritten title')
        plt.tight_layout()
        plt.show()
        
        # Check if x-tick labels are set
        assert len(axes[0].get_xticklabels()) == total_len
        assert axes[0].get_title() == 'Overwritten title'
        assert axes[0].get_xticklabels()[0].get_text() == 'Label_0'
        assert axes[0].get_xticklabels()[-1].get_text() == f'Label_{total_len - 1}'
        assert axes[1].get_xticklabels()[0].get_text() == 'Label_0'

    def test_with_labels_only_test(self, sample_data):
        """Test plotting with custom x-axis labels when only test data is plotted."""
        D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q = sample_data
        total_len = len(D_train) + len(D_test)
        labels = [f'Label_{i}' for i in range(total_len)]
        
        fig, axes = plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q, labels=labels, plot_train=False)
        plt.tight_layout()
        plt.show()
        
        # When plot_train is False, labels should be adjusted to test data length
        assert len(axes[0].get_xticklabels()) == len(D_test)
        assert axes[0].get_xticklabels()[0].get_text() == f'Label_{len(D_train)}' # First label should be from test set
        assert axes[1].get_xticklabels()[-1].get_text() == f'Label_{total_len - 1}'

    def test_with_event_index(self, sample_data):
        """Test highlighting specific event indices. Bars have increasing opacity"""
        D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q = sample_data
        event_indices = [5, len(D_train) + 2] # One in train, one in test
        opacity = np.linspace(0.01, 1, 30)

        fig, axes = plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q, 
                                event_index=event_indices, opacity=opacity)
        plt.tight_layout()
        plt.show()
        
        assert len(axes[0].patches) == 30 # 20 train and 10 test
        assert len(axes[1].patches) == 30 # 20 train and 10 test

    def test_logscale(self, sample_data):
        """Test plotting with logarithmic scale."""
        D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q = sample_data
        
        fig, axes = plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q, logscale=True)
        plt.tight_layout()
        plt.show()
        
        assert axes[0].get_yscale() == 'log'
        assert axes[1].get_yscale() == 'log'

    def test_scalar_thresholds(self, sample_data):
        """Test with scalar thresholds."""
        D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q = sample_data
        
        fig, axes = plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q)
        plt.tight_layout()
        plt.show()
        
        # Check for D threshold line
        d_threshold_lines = [line for line in axes[0].lines if line.get_label() == 'D threshold']
        assert len(d_threshold_lines) == 1
        assert d_threshold_lines[0].get_ydata()[0] == threshold_D

        # Check for Q threshold line
        q_threshold_lines = [line for line in axes[1].lines if line.get_label() == 'Q threshold']
        assert len(q_threshold_lines) == 1
        assert q_threshold_lines[0].get_ydata()[0] == threshold_Q

    def test_list_thresholds(self, sample_data):
        """Test with list thresholds and alpha values."""
        D_train, Q_train, D_test, Q_test, _, _ = sample_data
        threshold_D_list = [5.0, 8.0]
        threshold_Q_list = [2.0, 4.0]
        alpha_values = [0.01, 0.05]
        
        fig, axes = plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D_list, threshold_Q_list, alpha=alpha_values)
        plt.tight_layout()
        plt.show()
        
        # Check for D threshold lines
        d_threshold_lines = [line for line in axes[0].lines if 'D threshold' in line.get_label()]
        assert len(d_threshold_lines) == len(threshold_D_list)
        assert sorted([line.get_ydata()[0] for line in d_threshold_lines]) == sorted(threshold_D_list)

        # Check for Q threshold lines
        q_threshold_lines = [line for line in axes[1].lines if 'Q threshold' in line.get_label()]
        assert len(q_threshold_lines) == len(threshold_Q_list)
        assert sorted([line.get_ydata()[0] for line in q_threshold_lines]) == sorted(threshold_Q_list)

    def test_with_provided_axes(self, sample_data):
        """Test plotting on pre-existing matplotlib Axes objects."""
        D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q = sample_data
        
        # Create a figure and axes manually
        fig_manual, (ax_d_manual, ax_q_manual) = plt.subplots(2, 1)
        
        # Pass these axes to the function
        returned_fig, returned_axes = plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q, ax=[ax_d_manual, ax_q_manual])
        plt.tight_layout()
        plt.show()
        
        assert fig_manual == returned_fig
        assert returned_axes[0] is ax_d_manual
        assert returned_axes[1] is ax_q_manual
        
        # Check if content was added to the provided axes
        assert len(ax_d_manual.patches) > 0
        assert len(ax_q_manual.patches) > 0

    def test_invalid_ax_input(self):
        """Test that a ValueError is raised for invalid 'ax' input."""
        D_train = np.array([1])
        Q_train = np.array([1])
        D_test = np.array([1])
        Q_test = np.array([1])
        threshold_D = 1
        threshold_Q = 1

        # Test with a single ax instead of a list/tuple of two
        fig_single, ax_single = plt.subplots()
        with pytest.raises(ValueError, match="If 'ax' is provided, it must be a list or tuple of two matplotlib Axes objects."):
            plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q, ax=ax_single)
        plt.close(fig_single) # Close the figure created for the test

        # Test with a list of incorrect length
        fig_bad_len, ax_bad_len = plt.subplots(3, 1)
        with pytest.raises(ValueError, match="If 'ax' is provided, it must be a list or tuple of two matplotlib Axes objects."):
            plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q, ax=ax_bad_len)
        plt.close(fig_bad_len) # Close the figure created for the test

        # Test with non-list/tuple input
        with pytest.raises(ValueError, match="If 'ax' is provided, it must be a list or tuple of two matplotlib Axes objects."):
            plot_DQ_tt(D_train, Q_train, D_test, Q_test, threshold_D, threshold_Q, ax="invalid")