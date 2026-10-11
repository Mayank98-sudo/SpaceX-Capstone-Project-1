import unittest
from spacex_dash_app import app, get_pie_chart, get_scatter_chart, spacex_df

class DashboardTests(unittest.TestCase):
    def test_page_serves(self):
        client = app.server.test_client()
        self.assertEqual(client.get('/').status_code, 200)
        self.assertEqual(client.get('/_dash-layout').status_code, 200)

    def test_pie_matches_successes(self):
        figure = get_pie_chart('ALL')
        self.assertEqual(sum(figure.data[0].values), int(spacex_df['class'].sum()))

    def test_site_pie_includes_all_site_records(self):
        site = spacex_df['Launch Site'].iloc[0]
        self.assertEqual(len(get_pie_chart(site).data[0].labels), len(spacex_df[spacex_df['Launch Site'] == site]))

    def test_scatter_respects_site_and_payload(self):
        site = spacex_df['Launch Site'].iloc[0]
        expected = spacex_df[(spacex_df['Launch Site'] == site) & spacex_df['Payload Mass (kg)'].between(2000, 5000)]
        figure = get_scatter_chart(site, [2000, 5000])
        actual = sorted(float(v) for trace in figure.data for v in trace.x)
        self.assertEqual(actual, sorted(expected['Payload Mass (kg)'].astype(float)))

    def test_empty_payload_selection(self):
        self.assertEqual(len(get_scatter_chart('ALL', [-2, -1]).data), 0)

if __name__ == '__main__':
    unittest.main()
