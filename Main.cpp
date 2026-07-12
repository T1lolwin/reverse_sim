# include <Siv3D.hpp> // Siv3D v0.6.16
# include <fstream>
# include <random>

double Potential(double x) {
	return -0.5 * x * x + 0.25 * x * x * x * x;
}

double Alpha1(double x) {
	return x - x * x * x;
}

Vec2 ToScreen(double x, double y) {
	return Vec2{ Math::Map(x, -3.0, 3.0, 100.0, 900.0), Math::Map(y, -0.5, 2.0, 500.0, 100.0) };
}

class DiffusionSimulation {
private:
	// 定数パラメータ 
	const int32 num_grid = 500;
	const double D = 1.5;
	const double x_min = -3;
	const double x_max = 3;
	const double dx = (x_max - x_min) / (num_grid - 1);

	const double dt_pde = 0.0001;
	const int32 total_step = 518753;
	const int32 num_particles = 1000;
	const int32 save_interval = 500;
	const int32 total_frames = (total_step / save_interval) + 1;

	// データと状態 
	Array<float> s_data;
	Array<double> particle_history;
	AsyncTask<bool> calc_task;

	// アニメーション・UI用変数
	double acc_time = 0.0;
	double time_scale = 1.0;
	double current_frame_f = 0.0;
	bool is_playing = true;

	// フォントの用意
	const Font font_time{ 30, Typeface::Bold };
	const Font font_axis{ 14, Typeface::Regular };
	const Font font_loading{ 40, Typeface::Heavy };

	// gif
	AnimatedGIFWriter gifWriter;
	bool is_recording = false;

public:
	DiffusionSimulation() {
		const int32 total_element = total_step * num_grid;
		s_data.resize(total_element, 0.0f);

		std::ifstream ifs("score_data.bin", std::ios::binary);
		if (ifs) {
			ifs.read(reinterpret_cast<char*>(s_data.data()), total_element * sizeof(float));
		}
		else {
			System::MessageBoxOK(U"エラー", U"Appフォルダに score_data.bin が見つかりません。");
			System::Exit();
		}

		particle_history.resize(total_frames * num_particles);
		StartAsyncTask();
	}

	void Update() {
		if (!calc_task.isReady()) return;

		if (is_playing) {
			acc_time += Scene::DeltaTime() * time_scale;

			double frames_per_second = static_cast<double>(total_frames) / 10.0;

			current_frame_f = acc_time * frames_per_second;

			if (current_frame_f >= total_frames - 1) {
				acc_time = 0.0; // ループ再生
				current_frame_f = 0.0;
			}
		}

		UpdateUI();
	}

	void Draw() {
		if (!calc_task.isReady()) {
			DrawLoading();
			return;
		}

		int32 frame = static_cast<int32>(current_frame_f);
		int32 original_step = Min(frame * save_interval, total_step - 1);
		int32 draw_read_t = (total_step - 1) - original_step;

		DrawAxes();
		DrawVectorField(draw_read_t);
		DrawPotentialCurve();
		DrawHistogram(frame);
		DrawParticles(frame);
		DrawScoreForceArPeak(draw_read_t);
		DrawUIStatus(frame, original_step);
	}

private:
	void StartAsyncTask() {
		calc_task = Async([this]() {
			Array<double> particles(num_particles);
			std::normal_distribution<double> init_dist(1.0, 0.3);
			DefaultRNG rng;

			for (auto& p : particles) p = init_dist(rng);

			std::normal_distribution<double> noise_dist(0.0, Sqrt(D * dt_pde));
			int32 frame_cnt = 0;

			for (int32 step = 0; step < total_step; ++step) {
				// 履歴の保存
				if (step % save_interval == 0 && frame_cnt < total_frames) {
					for (int32 i = 0; i < num_particles; ++i) {
						particle_history[frame_cnt * num_particles + i] = particles[i];
					}
					frame_cnt++;
				}

				int32 read_t = (total_step - 1) - step;

				for (auto& p : particles) {
					p = Clamp(p, x_min, x_max);

					// 【線形補間の実装】粒子の現在地 p が、格子点のどこにあるかを計算
					double grid_pos = (p - x_min) / dx;
					int32 idx_left = static_cast<int32>(grid_pos);
					int32 idx_right = Min(idx_left + 1, num_grid - 1);
					double weight_right = grid_pos - idx_left;

					float s_left = s_data[read_t * num_grid + idx_left];
					float s_right = s_data[read_t * num_grid + idx_right];
					double s_interpolated = s_left * (1.0 - weight_right) + s_right * weight_right;

					double drift = -(Alpha1(p) - D * s_interpolated);
					double diffusion = noise_dist(GetDefaultRNG());

					p += drift * dt_pde + diffusion;
				}
			}
			return true;
	});
	}

	void DrawLoading() {
		Circle{ 500, 300, 50 }.drawArc(Scene::Time() * 180_deg, 90_deg, 5, 5, Palette::Skyblue);
		font_loading(U"軌跡を計算中...").drawAt(500, 400, Palette::White);
	}

	void DrawAxes() const {
		double axisY = 500.0;
		Line{ 100, axisY, 900, axisY }.draw(1.0, ColorF{ 0.4 });

		for (int32 val = x_min; val <= x_max; ++val) {
			double screenX = Math::Map(val, x_min, x_max, 100, 900);
			Line{ screenX, axisY, screenX, axisY + 5 }.draw(1.0, ColorF{ 0.6 });
			font_axis(Format(val)).drawAt(screenX, axisY + 20, ColorF{ 0.7 });
		}
	}

	void DrawVectorField(int32 draw_read_t) const {
		const int32 x_skip = 15;
		const int32 y_step = 25;

		for (int32 i = 0; i < num_grid; i += x_skip) {
			double x_val = x_min + i * dx;
			double screenX = Math::Map(x_val, x_min, x_max, 100.0, 900.0);

			float s_val = s_data[draw_read_t * num_grid + i];
			double force = D * s_val;

			if (Abs(force) < 0.2) continue;

			double arrow_length = Clamp(force * 10.0, -35.0, 35.0);

			double alpha = Clamp(Abs(force) / 10.0, 0.1, 0.4);
			ColorF arrow_color = force > 0 ? ColorF{ 1.0, 0.3, 0.3, alpha } : ColorF{ 0.3, 0.6, 1.0, alpha };

			for (int32 screenY = 60; screenY < 500; screenY += y_step) {
				Vec2 start{ screenX - arrow_length / 2, screenY };
				Vec2 end{ screenX + arrow_length / 2, screenY };
				Line{ start, end }.drawArrow(2.0, Vec2{ 6, 6 }, arrow_color);
			}
		}
	}

	void DrawPotentialCurve() const {
		for (int32 i = 0; i < num_grid - 1; ++i) {
			double x1 = x_min + i * dx;
			double x2 = x_min + (i + 1) * dx;
			Line{ ToScreen(x1, Potential(x1)), ToScreen(x2, Potential(x2)) }.draw(4.0, ColorF{ 1.0, 1.0, 1.0, 0.9 });
		}
	}

	void DrawHistogram(int32 frame) const {
		// 確率分布（ヒストグラム）の計算と描画
		const int32 num_bins = 100; // 空間を100個の区間に分ける
		Array<int32> bins(num_bins, 0);

		// 現在のコマの粒子の位置を集計する
		for (int32 i = 0; i < num_particles; ++i) {
			double p_x = particle_history[frame * num_particles + i];
			int32 b = static_cast<int32>(Math::Map(p_x, x_min, x_max, 0, num_bins - 1));
			bins[Clamp(b, 0, num_bins - 1)]++;
		}

		// 集計結果をポテンシャル曲線の上に棒グラフとして描く
		for (int32 i = 0; i < num_bins; ++i) {
			if (bins[i] == 0) continue; // 粒子がいない場所は描かない

			double bin_x_center = Math::Map(i + 0.5, 0, num_bins, x_min, x_max);
			Vec2 base_pos = ToScreen(bin_x_center, Potential(bin_x_center));

			double width = 800.0 / num_bins;
			double height = bins[i] * 1.5; // グラフの高さ倍率

			// 半透明の青緑色でヒストグラムを描画
			RectF{ Arg::bottomCenter = base_pos, width * 0.9, height }.draw(ColorF{ 0.2, 0.8, 0.6, 0.5 });
		}
	}

	void DrawParticles(int32 frame) const {
		for (int32 i = 0; i < num_particles; ++i) {
			double p_x = particle_history[frame * num_particles + i];
			Vec2 pos = ToScreen(p_x, Potential(p_x));

			ColorF p_color = (Abs(p_x) < 0.2) ? ColorF{ 1.0, 0.2, 0.8, 0.9 } : ColorF{ 0.0, 0.8, 1.0, 0.7 };
			Circle{ pos, 4.0 }.draw(p_color);
		}
	}

	void DrawScoreForceArPeak(int32 draw_read_t) const {
		double target_x = 0.0;
		double grid_pos_zero = (target_x - x_min) / dx;
		int32 idx_left = static_cast<int32>(grid_pos_zero);
		int32 idx_right = Min(idx_left + 1, num_grid - 1);
		double weight_right = grid_pos_zero - idx_left;

		float s_left = s_data[draw_read_t * num_grid + idx_left];
		float s_right = s_data[draw_read_t * num_grid + idx_right];

		// x=0 の両隣にある格子点のデータから、ピッタリ x=0.0 のスコアを割り出す
		double s_center = s_left * (1.0 - weight_right) + s_right * weight_right;

		double force_at_peak = D * s_center;

		Vec2 peak_pos = ToScreen(0.0, Potential(0.0));

		String force_text = force_at_peak > 0 ? U"右へ引く力: {:.2f}"_fmt(force_at_peak) : U"左へ引く力: {:.2f}"_fmt(Abs(force_at_peak));
		ColorF force_color = force_at_peak > 0 ? ColorF{ 1.0, 0.4, 0.4 } : ColorF{ 0.4, 0.6, 1.0 };

		font_axis(force_text).drawAt(peak_pos.x, peak_pos.y - 40, force_color);

		double arrow_length = Clamp(force_at_peak * 20.0, -150.0, 150.0);
		if (Abs(arrow_length) > 2.0) {
			Line{ peak_pos.x, peak_pos.y - 20, peak_pos.x + arrow_length, peak_pos.y - 20 }
			.drawArrow(3.0, Vec2{ 10.0, 10.0 }, force_color);
		}
	}

	void DrawUIStatus(int32 frame, int32 original_step) const {
		int32 left_count = 0;

		for (int32 i = 0; i < num_particles; ++i) {
			if (particle_history[frame * num_particles + i] < 0.0) left_count++;
		}
		double left_ratio = static_cast<double>(left_count) / num_particles;
		double right_ratio = 1 - left_ratio;

		RectF{ 700, 30, 250, 20 }.draw(ColorF{ 0.2 });
		RectF{ 700, 30, 250 * left_ratio, 20 }.draw(ColorF{ 0.3, 0.5, 1.0 });

		font_axis(U"左の井戸: {:.1f}%"_fmt(left_ratio * 100)).draw(700, 55, Palette::White);
		font_axis(U"右の井戸: {:.1f}%"_fmt(right_ratio * 100)).draw(850, 55, Palette::White);

		double real_time = Math::Map(original_step, 0, total_step, 10.0, 0.0);
		font_time(U"Time t = {:.2f}"_fmt(real_time)).draw(50, 30, Palette::White);
	}

	void UpdateUI() {
		if (SimpleGUI::Button(is_playing ? U"⏸" : U"▶", Vec2{ 50, 570 }, 50)) {
			is_playing = !is_playing;
		}

		SimpleGUI::Slider(U"再生速度: {:.1f}x"_fmt(time_scale), time_scale, 0.1, 3.0, Vec2{ 200, 610 }, 150, 550);

		double slider_val = current_frame_f;

		if (SimpleGUI::Slider(U"タイムライン", slider_val, 0.0, static_cast<double>(total_frames - 1), Vec2{ 200, 570 }, 150, 550)) {
			is_playing = false; // 操作中は一時停止
			current_frame_f = slider_val;

			double frames_per_second = static_cast<double>(total_frames) / 10.0;
			acc_time = current_frame_f / frames_per_second; // シークバーの位置に合わせて時間を更新
		}
	}
};

void Main() {
	Window::Resize(1000, 650);
	Scene::SetBackground(ColorF{ 0.07, 0.08, 0.14 });

	DiffusionSimulation sim;

	while (System::Update()) {
		sim.Update();
		sim.Draw();
	}
}

	

//
// - Debug ビルド: プログラムの最適化を減らす代わりに、エラーやクラッシュ時に詳細な情報を得られます。
//
// - Release ビルド: 最大限の最適化でビルドします。
//
// - [デバッグ] メニュー → [デバッグの開始] でプログラムを実行すると、[出力] ウィンドウに詳細なログが表示され、エラーの原因を探せます。
//
// - Visual Studio を更新した直後は、プログラムのリビルド（[ビルド]メニュー → [ソリューションのリビルド]）が必要な場合があります。
//
