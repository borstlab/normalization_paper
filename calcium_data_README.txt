

calcium_data_fig3.pkl
------------------------------------------------
This file contains a serialized pandas-Dataframe for the data in Figure 3.

Open with:

>> pandas.read_pickle('calcium_data_fig3.pkl')

The column "time" is the relative time for each stimulus trial with zero defined as the time point when the stimulus started moving.

The indices are:

>> "fg_contrast" : foreground contrast
>> "bg_contrast" : background contrast
>> "cell_type"   : L1, L2, L3, L4, L5, Mi1, Tm3, Mi4, Mi9, Tm1, Tm2, Tm3, Tm4, T4, T5
>> "name"        : individual name given to each ROI

calcium_data_fig4_abc.pkl
------------------------------------------------
This file contains a serialized pandas-Dataframe for the data in Figure 4 a,b,c.

Open with:

>> pandas.read_pickle('calcium_data_fig4_abc.pkl')

The column "time" is the relative time for each stimulus trial with zero defined as the time point when the stimulus started moving.

The indices are:

>> "stimulus_name" : either "direction_tuning", "frequency_tuning", "size_tuning" or "reference stimulus", corresponding to the panels a,b,c or the reference stimulus (0% background contrast, black dashed line in a,b,c)
>> "direction" : motion direction, if "stimulus_name" is "direction_tuning"
>> "contrast_frequency" : background frequency, if "stimulus_name" is "frequency_tuning"
>> "annulus_diameter" : annulus diameter, if "stimulus_name" is "size_tuning"
>> "cell_type"   : Mi1, Tm3, Tm1, Tm2
>> "name"        : individual name given to each ROI


calcium_data_fig4_d.pkl
------------------------------------------------
This file contains a serialized pandas-Dataframe for the data in Figure 4 d.

Open with:

>> pandas.read_pickle('calcium_data_fig4_d.pkl')

The column "time" is the relative time for each stimulus trial with zero defined as the time point when the center dot flashed.

The indices are:

>> "grating_masked" : False, if static grating is presented before, True, if grating is masked before
>> "interval" : time interval between grating motion onset and dot flash in seconds
>> "cell_type"   : Tm3, Tm2split
>> "name"        : individual name given to each ROI



calcium_data_fig5_dfhj.pkl
------------------------------------------------
This file contains a serialized pandas-Dataframe for the data in Figure 5 d,f,h,j.

Open with:

>> pandas.read_pickle('calcium_data_fig5_dfhj.pkl')

The column "time" is the relative time for each stimulus trial with zero defined as the time point when the stimulus started moving.

Contrast experiment with background contrast either 0% or 100%. Same structure as data_fig3.pkl.

The indices are:

>> "fg_contrast" : foreground contrast
>> "bg_contrast" : background contrast (only 0% or 100%)
>> "cell_type"   : Mi1TNT, Mi1TNTin, Tm1TNT, Tm1TNTin, Tm2splitTNT, Tm2splitTNTin, Tm3TNT, Tm2TNTin
>> "name"        : individual name given to each ROI


calcium_data_fig5_cegi.pkl
------------------------------------------------
This file contains a serialized pandas-Dataframe for the data in Figure 5 c,e,g,i.

Open with:

>> pandas.read_pickle('calcium_data_fig5_cegi.pkl')

The column "time" is the relative time for each stimulus trial with zero defined as the time point when the stimulus started moving.

Frequency tuning experiment as in Figure 4 b. Similar structure to data_fig4_abc.pkl.

The indices are:

>> "stimulus_name" : either "frequency_tuning" or "reference stimulus"
>> "contrast_frequency" : background frequency, if "stimulus_name" is "frequency_tuning"
>> "cell_type"   : Mi1TNT, Mi1TNTin, Tm1TNT, Tm1TNTin, Tm2splitTNT, Tm2splitTNTin, Tm3TNT, Tm2TNTin
>> "name"        : individual name given to each ROI


