


		

<!DOCTYPE html>
<html lang="en-US" >

<head>

	
	<script>
window.ts_endpoint_url = "https:\/\/slack.com\/beacon\/timing";

(function(e) {
	var n=Date.now?Date.now():+new Date,r=e.performance||{},t=[],a={},i=function(e,n){for(var r=0,a=t.length,i=[];a>r;r++)t[r][e]==n&&i.push(t[r]);return i},o=function(e,n){for(var r,a=t.length;a--;)r=t[a],r.entryType!=e||void 0!==n&&r.name!=n||t.splice(a,1)};r.now||(r.now=r.webkitNow||r.mozNow||r.msNow||function(){return(Date.now?Date.now():+new Date)-n}),r.mark||(r.mark=r.webkitMark||function(e){var n={name:e,entryType:"mark",startTime:r.now(),duration:0};t.push(n),a[e]=n}),r.measure||(r.measure=r.webkitMeasure||function(e,n,r){n=a[n].startTime,r=a[r].startTime,t.push({name:e,entryType:"measure",startTime:n,duration:r-n})}),r.getEntriesByType||(r.getEntriesByType=r.webkitGetEntriesByType||function(e){return i("entryType",e)}),r.getEntriesByName||(r.getEntriesByName=r.webkitGetEntriesByName||function(e){return i("name",e)}),r.clearMarks||(r.clearMarks=r.webkitClearMarks||function(e){o("mark",e)}),r.clearMeasures||(r.clearMeasures=r.webkitClearMeasures||function(e){o("measure",e)}),e.performance=r,"function"==typeof define&&(define.amd||define.ajs)&&define("performance",[],function(){return r}) // eslint-disable-line
})(window);

</script>
<script>

;(function() {
'use strict';


window.TSMark = function(mark_label) {
	if (!window.performance || !window.performance.mark) return;
	performance.mark(mark_label);
};
window.TSMark('start_load');


window.TSMeasureAndBeacon = function(measure_label, start_mark_label) {
	if (start_mark_label === 'start_nav' && window.performance && window.performance.timing) {
		window.TSBeacon(measure_label, (new Date()).getTime() - performance.timing.navigationStart);
		return;
	}
	if (!window.performance || !window.performance.mark || !window.performance.measure) return;
	performance.mark(start_mark_label + '_end');
	try {
		performance.measure(measure_label, start_mark_label, start_mark_label + '_end');
		window.TSBeacon(measure_label, performance.getEntriesByName(measure_label)[0].duration);
	} catch (e) {
		
	}
};


if ('sendBeacon' in navigator) {
	window.TSBeacon = function(label, value) {
		var endpoint_url = window.ts_endpoint_url || 'https://slack.com/beacon/timing';
		navigator.sendBeacon(endpoint_url + '?data=' + encodeURIComponent(label + ':' + value), '');
	};
} else {
	window.TSBeacon = function(label, value) {
		var endpoint_url = window.ts_endpoint_url || 'https://slack.com/beacon/timing';
		(new Image()).src = endpoint_url + '?data=' + encodeURIComponent(label + ':' + value);
	};
}
})();
</script>
 

<script>
window.TSMark('step_load');
</script>	<noscript><meta http-equiv="refresh" content="0; URL=/files/jzhang/F6DMT5DMJ/viz.py?nojsmode=1" /></noscript>
<script type="text/javascript">
if(self!==top)window.document.write("\u003Cstyle>body * {display:none !important;}\u003C\/style>\u003Ca href=\"#\" onclick="+
"\"top.location.href=window.location.href\" style=\"display:block !important;padding:10px\">Go to Slack.com\u003C\/a>");

(function() {
	var timer;
	if (self !== top) {
		timer = window.setInterval(function() {
			if (window.$) {
				try {
					$('#page').remove();
					$('#client-ui').remove();
					window.TS = null;
					window.clearInterval(timer);
				} catch(e) {}
			}
		}, 200);
	}
}());

</script>

<script>(function() {
	'use strict';

	window.callSlackAPIUnauthed = function(method, args, callback) {
		var timestamp = Date.now() / 1000;  
		var version = (window.TS && TS.boot_data && TS.boot_data.version_uid) ? TS.boot_data.version_uid.substring(0, 8) : 'noversion';
		var url = '/api/' + method + '?_x_id=' + version + '-' + timestamp;
		var req = new XMLHttpRequest();

		req.onreadystatechange = function() {
			if (req.readyState == 4) {
				req.onreadystatechange = null;
				var obj;

				if (req.status == 200 || req.status == 429) {
					try {
						obj = JSON.parse(req.responseText);
					} catch (err) {
						TS.console.warn(8675309, 'unable to do anything with api rsp');
					}
				}

				obj = obj || {
					ok: false,
				};

				callback(obj.ok, obj, args);
			}
		};

		var async = true;
		req.open('POST', url, async);

		var form_data = new FormData();
		var has_data = false;
		Object.keys(args).forEach(function(k) {
			if (k[0] === '_') return;
			form_data.append(k, args[k]);
			has_data = true;
		});

		if (has_data) {
			req.send(form_data);
		} else {
			req.send();
		}
	};
})();
</script>

	<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/webpack.manifest.b79cd92e5efb0aed884a.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>

			
	
		<script>
			if (window.location.host == 'slack.com' && window.location.search.indexOf('story') < 0) {
				document.cookie = '__cvo_skip_doc=' + escape(document.URL) + '|' + escape(document.referrer) + ';path=/';
			}
		</script>
	

		<script type="text/javascript">
		
		try {
			if(window.location.hash && !window.location.hash.match(/^(#?[a-zA-Z0-9_]*)$/)) {
				window.location.hash = '';
			}
		} catch(e) {}
		
	</script>

	<script type="text/javascript">
				
			window.optimizely = [];
			window.dataLayer = [];
			window.ga = false;
		
	
				(function(e,c,b,f,d,g,a){e.SlackBeaconObject=d;
		e[d]=e[d]||function(){(e[d].q=e[d].q||[]).push([1*new Date(),arguments])};
		e[d].l=1*new Date();g=c.createElement(b);a=c.getElementsByTagName(b)[0];
		g.async=1;g.src=f;a.parentNode.insertBefore(g,a)
		})(window,document,"script","https://a.slack-edge.com/bv1-1/slack_beacon.a3004ddafd9de6722fc4.min.js","sb");
		sb('set', 'token', '3307f436963e02d4f9eb85ce5159744c');

					sb('set', 'user_id', "U1V3NSQ6N");
							sb('set', 'user_' + "batch", "signup_api");
							sb('set', 'user_' + "created", "2016-07-25");
						sb('set', 'name_tag', "ucbischool" + '/' + "arvin.sahni");
				sb('track', 'pageview');

		
		function track(a) {
			if(window.ga) ga('send','event','web', a);
			if(window.sb) sb('track', a);
		}
		

	</script>

	
	<meta name="referrer" content="no-referrer">
		<meta name="superfish" content="nofish">

	<script type="text/javascript">



var TS_last_log_date = null;
var TSMakeLogDate = function() {
	var date = new Date();

	var y = date.getFullYear();
	var mo = date.getMonth()+1;
	var d = date.getDate();

	var time = {
	  h: date.getHours(),
	  mi: date.getMinutes(),
	  s: date.getSeconds(),
	  ms: date.getMilliseconds()
	};

	Object.keys(time).map(function(moment, index) {
		if (moment == 'ms') {
			if (time[moment] < 10) {
				time[moment] = time[moment]+'00';
			} else if (time[moment] < 100) {
				time[moment] = time[moment]+'0';
			}
		} else if (time[moment] < 10) {
			time[moment] = '0' + time[moment];
		}
	});

	var str = y + '/' + mo + '/' + d + ' ' + time.h + ':' + time.mi + ':' + time.s + '.' + time.ms;
	if (TS_last_log_date) {
		var diff = date-TS_last_log_date;
		//str+= ' ('+diff+'ms)';
	}
	TS_last_log_date = date;
	return str+' ';
}

var parseDeepLinkRequest = function(code) {
	var m = code.match(/"id":"([CDG][A-Z0-9]{8})"/);
	var id = m ? m[1] : null;

	m = code.match(/"team":"(T[A-Z0-9]{8})"/);
	var team = m ? m[1] : null;

	m = code.match(/"message":"([0-9]+\.[0-9]+)"/);
	var message = m ? m[1] : null;

	return { id: id, team: team, message: message };
}

if ('rendererEvalAsync' in window) {
	var origRendererEvalAsync = window.rendererEvalAsync;
	window.rendererEvalAsync = function(blob) {
		try {
			var data = JSON.parse(decodeURIComponent(atob(blob)));
			if (data.code.match(/handleDeepLink/)) {
				var request = parseDeepLinkRequest(data.code);
				if (!request.id || !request.team || !request.message) return;

				request.cmd = 'channel';
				TSSSB.handleDeepLinkWithArgs(JSON.stringify(request));
				return;
			} else {
				origRendererEvalAsync(blob);
			}
		} catch (e) {
		}
	}
}
</script>



<script type="text/javascript">

	var TSSSB = {
		call: function() {
			return false;
		}
	};

</script>
<script>TSSSB.env = (function() {
	'use strict';

	var v = {
		win_ssb_version: null,
		win_ssb_version_minor: null,
		mac_ssb_version: null,
		mac_ssb_version_minor: null,
		mac_ssb_build: null,
		lin_ssb_version: null,
		lin_ssb_version_minor: null,
		desktop_app_version: null,
	};

	var is_win = (navigator.appVersion.indexOf('Windows') !== -1);
	var is_lin = (navigator.appVersion.indexOf('Linux') !== -1);
	var is_mac = !!(navigator.userAgent.match(/(OS X)/g));

	if (navigator.userAgent.match(/(Slack_SSB)/g) || navigator.userAgent.match(/(Slack_WINSSB)/g)) {
		
		var parts = navigator.userAgent.split('/');
		var version_str = parts[parts.length - 1];
		var version_float = parseFloat(version_str);
		var version_parts = version_str.split('.');
		var version_minor = (version_parts.length == 3) ? parseInt(version_parts[2], 10) : 0;

		if (navigator.userAgent.match(/(AtomShell)/g)) {
			
			if (is_lin) {
				v.lin_ssb_version = version_float;
				v.lin_ssb_version_minor = version_minor;
			} else if (is_win) {
				v.win_ssb_version = version_float;
				v.win_ssb_version_minor = version_minor;
			} else if (is_mac) {
				v.mac_ssb_version = version_float;
				v.mac_ssb_version_minor = version_minor;
			}

			if (version_parts.length >= 3) {
				v.desktop_app_version = {
					major: parseInt(version_parts[0], 10),
					minor: parseInt(version_parts[1], 10),
					patch: parseInt(version_parts[2], 10),
				};
			}
		} else {
			
			v.mac_ssb_version = version_float;
			v.mac_ssb_version_minor = version_minor;

			
			
			var app_ver = window.macgap && macgap.app && macgap.app.buildVersion && macgap.app.buildVersion();
			var matches = String(app_ver).match(/(?:\()(.*)(?:\))/);
			v.mac_ssb_build = (matches && matches.length == 2) ? parseInt(matches[1] || 0, 10) : 0;
		}
	}

	return v;
})();
</script>


	<script type="text/javascript">
		
		window.addEventListener('load', function() {
			var was_TS = window.TS;
			delete window.TS;
			if (was_TS) window.TS = was_TS;
		});
	</script>
	    <title>viz.py | UC Berkeley School of Information Slack</title>
    <meta name="author" content="Slack">

	
		
	
	
		
	
				
	
	

	
	
	
	
	
	
	
			<!-- output_css "core" -->
    <link href="https://a.slack-edge.com/f852a/style/rollup-plastic.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">

		<!-- output_css "before_file_pages" -->
    <link href="https://a.slack-edge.com/74a30/style/libs/codemirror.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">
    <link href="https://a.slack-edge.com/d3447/style/codemirror_overrides.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">

	<!-- output_css "file_pages" -->
    <link href="https://a.slack-edge.com/e3786/style/rollup-file_pages.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">

	
			<!-- output_css "regular" -->
    <link href="https://a.slack-edge.com/c2e9/style/print.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">
    <link href="https://a.slack-edge.com/96b46/style/libs/quill.core.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">
    <link href="https://a.slack-edge.com/45791/style/texty.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">
    <link href="https://a.slack-edge.com/b003/style/libs/lato-2-compressed.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">
    <link href="https://a.slack-edge.com/98897/style/_helpers.css" rel="stylesheet" type="text/css" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)">

	

	
	
		<meta name="robots" content="noindex, nofollow" />
	

	
<link id="favicon" rel="shortcut icon" href="https://a.slack-edge.com/bc86/marketing/img/meta/favicon-32.png" sizes="16x16 32x32 48x48" type="image/png" />

<link rel="icon" href="https://a.slack-edge.com/bc86/marketing/img/meta/app-256.png" sizes="256x256" type="image/png" />

<link rel="apple-touch-icon-precomposed" sizes="152x152" href="https://a.slack-edge.com/bc86/marketing/img/meta/ios-152.png" />
<link rel="apple-touch-icon-precomposed" sizes="144x144" href="https://a.slack-edge.com/bc86/marketing/img/meta/ios-144.png" />
<link rel="apple-touch-icon-precomposed" sizes="120x120" href="https://a.slack-edge.com/bc86/marketing/img/meta/ios-120.png" />
<link rel="apple-touch-icon-precomposed" sizes="114x114" href="https://a.slack-edge.com/bc86/marketing/img/meta/ios-114.png" />
<link rel="apple-touch-icon-precomposed" sizes="72x72" href="https://a.slack-edge.com/bc86/marketing/img/meta/ios-72.png" />
<link rel="apple-touch-icon-precomposed" href="https://a.slack-edge.com/bc86/marketing/img/meta/ios-57.png" />

<meta name="msapplication-TileColor" content="#FFFFFF" />
<meta name="msapplication-TileImage" content="https://a.slack-edge.com/bc86/marketing/img/meta/app-144.png" />
	
	<!--[if lt IE 9]>
	<script src="https://a.slack-edge.com/bv1-1/html5shiv.4b476930dc209b36fdb2.min.js"></script>
	<![endif]-->

</head>

<body class="						">

		  			<script>
		
			var w = Math.max(document.documentElement.clientWidth, window.innerWidth || 0);
			if (w > 1440) document.querySelector('body').classList.add('widescreen');
		
		</script>
	
  	
	

			

<nav id="site_nav" class="no_transition">

	<div id="site_nav_contents">

		<div id="user_menu">
			<div id="user_menu_contents">
				<div id="user_menu_avatar">
										<span class="member_image thumb_48" style="background-image: url('https://avatars.slack-edge.com/2017-01-12/127270365911_9d544bb6f6d96cbac129_192.jpg')" data-thumb-size="48" data-member-id="U1V3NSQ6N"></span>
					<span class="member_image thumb_36" style="background-image: url('https://avatars.slack-edge.com/2017-01-12/127270365911_9d544bb6f6d96cbac129_72.jpg')" data-thumb-size="36" data-member-id="U1V3NSQ6N"></span>
				</div>
				<h3>Signed in as</h3>
				<span id="user_menu_name">arvin.sahni</span>
			</div>
		</div>

		<div class="nav_contents">

			<ul class="primary_nav">
									<li><a href="/messages" data-qa="app"><i class="ts_icon ts_icon_angle_arrow_up_left"></i>Back to Slack</a></li>
								<li><a href="/home" data-qa="home"><i class="ts_icon ts_icon_home"></i>Home</a></li>
				<li><a href="/account" data-qa="account_profile"><i class="ts_icon ts_icon_user"></i>Account &amp; Profile</a></li>
				<li><a href="/apps/manage" data-qa="configure_apps" target="_blank"><i class="ts_icon ts_icon_plug"></i>Configure Apps</a></li>
				<li><a href="/files" data-qa="files"><i class="ts_icon ts_icon_all_files clear_blue"></i>Files</a></li>
									<li><a href="/team" data-qa="team_directory"><i class="ts_icon ts_icon_team_directory"></i>Directory</a></li>
																									<li><a href="/stats" data-qa="statistics"><i class="ts_icon ts_icon_dashboard"></i>Analytics</a></li>
																		<li><a href="/customize" data-qa="customize"><i class="ts_icon ts_icon_magic"></i>Customize</a></li>
													<li><a href="/account/team" data-qa="team_settings"><i class="ts_icon ts_icon_cog_o"></i>Team Settings</a></li>
							</ul>

			
		</div>

		<div id="footer">

			<ul id="footer_nav">
				<li><a href="/is" data-qa="tour">Tour</a></li>
				<li><a href="/downloads" data-qa="download_apps">Download Apps</a></li>
				<li><a href="/brand-guidelines" data-qa="brand_guidelines">Brand Guidelines</a></li>
				<li><a href="/help" data-qa="help">Help</a></li>
				<li><a href="https://api.slack.com" target="_blank" data-qa="api">API<i class="ts_icon ts_icon_external_link small_left_margin ts_icon_inherit"></i></a></li>
								<li><a href="/pricing?ui_step=96&ui_element=5" data-qa="pricing" data-clog-event="GROWTH_PRICING" data-clog-ui-element="pricing_link" data-clog-ui-action="click" data-clog-ui-step="admin_footer">Pricing</a></li>
				<li><a href="/help/requests/new" data-qa="contact">Contact</a></li>
				<li><a href="/terms-of-service" data-qa="policies">Policies</a></li>
				<li><a href="http://slackhq.com/" target="_blank" data-qa="our_blog">Our Blog</a></li>
				<li><a href="https://slack.com/signout/30345778662?crumb=s-1501032381-48a0e9f9c8-%E2%98%83" data-qa="sign_out">Sign Out<i class="ts_icon ts_icon_sign_out small_left_margin ts_icon_inherit"></i></a></li>
			</ul>

			<p id="footer_signature">Made with <i class="ts_icon ts_icon_heart"></i> by Slack</p>

		</div>

	</div>
</nav>	
			
<header>
			<a id="menu_toggle" class="no_transition" data-qa="menu_toggle_hamburger">
			<span class="menu_icon"></span>
			<span class="menu_label">Menu</span>
			<span class="vert_divider"></span>
		</a>
		<h1 id="header_team_name" class="inline_block no_transition" data-qa="header_team_name">
			<a href="/home">
				<i class="ts_icon ts_icon_home" /></i>
				UC Berkeley School of Information
			</a>
		</h1>
		<div class="header_nav">
			<div class="header_btns float_right">
				<a id="team_switcher" data-qa="team_switcher">
					<i class="ts_icon ts_icon_th_large ts_icon_inherit"></i>
					<span class="block label">Teams</span>
				</a>
				<a href="/help" id="help_link" data-qa="help_link">
					<i class="ts_icon ts_icon_life_ring ts_icon_inherit"></i>
					<span class="block label">Help</span>
				</a>
															<a href="/messages" data-qa="launch">
							<img src="https://a.slack-edge.com/66f9/img/icons/ios-64.png" srcset="https://a.slack-edge.com/66f9/img/icons/ios-32.png 1x, https://a.slack-edge.com/66f9/img/icons/ios-64.png 2x" title="Slack" alt="Launch Slack"/>
							<span class="block label">Launch</span>
						</a>
												</div>
				                    <ul id="header_team_nav" data-qa="team_switcher_menu">
	                        	                            <li class="active">
	                            	<a href="https://ucbischool.slack.com/home" target="https://ucbischool.slack.com/">
	                            			                            			<i class="ts_icon small ts_icon_check_circle_o active_icon s"></i>
	                            			                            				                            		<i class="team_icon small" style="background-image: url('https://s3-us-west-2.amazonaws.com/slack-files2/avatars/2017-01-12/127065444132_1a5231f93eb2d55cf835_88.png');"></i>
		                            		                            		<span class="switcher_label team_name">UC Berkeley School of Information</span>
	                            	</a>
	                            </li>
	                        	                            <li >
	                            	<a href="https://pugs-org-sg.slack.com/home" target="https://pugs-org-sg.slack.com/">
	                            			                            				                            		<i class="team_icon small" style="background-image: url('https://s3-us-west-2.amazonaws.com/slack-files2/avatars/2015-08-01/8492255008_bb25b8d20bc84b9d7aa2_88.jpg');"></i>
		                            		                            		<span class="switcher_label team_name">Python User Group Singapore</span>
	                            	</a>
	                            </li>
	                        	                        <li id="add_team_option"><a href="https://slack.com/signin" target="_blank"><i class="ts_icon ts_icon_plus team_icon small"></i> <span class="switcher_label">Sign in to another team...</span></a></li>
	                    </ul>
	                		</div>
	
	
	
</header>	
	<div id="page" >

		<div id="page_contents" data-qa="page_contents" class="">


<p class="print_only">
	<strong>
	
	Created by jzhang on Wednesday, July 26, 2017 at 9:23 AM
	</strong><br />
	<span class="subtle_silver break_word">https://ucbischool.slack.com/files/jzhang/F6DMT5DMJ/viz.py</span>
</p>

<div class="file_header_container no_print"></div>

<div class="alert_container">
		

<div class="file_public_link_shared alert" style="display: none;">
		
	<i class="ts_icon ts_icon_link"></i> Public Link: <a class="file_public_link" href="https://slack-files.com/T0WA5NWKG-F6DMT5DMJ-ed1b6c314e" target="new">https://slack-files.com/T0WA5NWKG-F6DMT5DMJ-ed1b6c314e</a>
</div></div>

<div id="file_page" class="card top_padding">

	<p class="small subtle_silver no_print meta">
		
		4KB Python snippet created on <span class="date">July 26, 2017</span>.
				This file is private.		<span class="file_share_list"></span>
	</p>

	<a id="file_action_cog" class="action_cog action_cog_snippet float_right no_print">
		<span>Actions </span><i class="ts_icon ts_icon_cog"></i>
	</a>
	<a id="snippet_expand_toggle" class="float_right no_print">
		<i class="ts_icon ts_icon_expand "></i>
		<i class="ts_icon ts_icon_compress hidden"></i>
	</a>

	<div class="large_bottom_margin clearfix">
		<pre id="file_contents">from __future__ import division

from flask import render_template, request, Response, jsonify, send_from_directory
from app import app

import json
import psycopg2
import os
import sys
import psycopg2.extras
import pandas as pd

module_path = os.path.abspath(os.path.join(&#039;../&#039;))
if module_path not in sys.path:
    sys.path.append(module_path)

from learn import forall as fa
from learn import utils

@app.route(&#039;/index&#039;)
def index():
    return render_template(&#039;home.html&#039;)


@app.route(&#039;/viz&#039;)
def viz():
    return render_template(&#039;viz.html&#039;)


def to_csv(d, fields):
    d.insert(0, fields)
    return Response(&#039;\n&#039;.join([&quot;,&quot;.join(map(str, e)) for e in d]), mimetype=&#039;text/csv&#039;)


@app.route(&#039;/hist_data&#039;, methods=[&#039;GET&#039;, &#039;POST&#039;])
def hist_data():
    website = request.args.get(&#039;website&#039;)
    person = request.args.get(&#039;person&#039;)
    db = psycopg2.connect(host=&#039;ec2-54-208-219-223.compute-1.amazonaws.com&#039;,
                          database=&#039;election&#039;,
                          user=&#039;elections&#039;,
                          password=&#039;election2016&#039;)
    curs = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    DEC2FLOAT = psycopg2.extensions.new_type(
        psycopg2.extensions.DECIMAL.values,
        &#039;DEC2FLOAT&#039;,
        lambda value, curs: float(value) if value is not None else None)
    psycopg2.extensions.register_type(DEC2FLOAT)

    if website:
        sql = &quot;&quot;&quot;select a.bin, sum(coalesce(count,0)) from histogram_bins a
                left join (select * from data_binned where website = &#039;%s&#039; and person = &#039;%s&#039;) b on a.bin = b.bin
                group by 1 order by 1&quot;&quot;&quot; % (website, person)
    else:
        sql = &quot;&quot;&quot;select a.bin, sum(coalesce(count,0)) from histogram_bins a
                left join (select * from data_binned where person = &#039;%s&#039;) b on a.bin = b.bin
                group by 1 order by 1&quot;&quot;&quot; % person
    print(sql)
    curs.execute(sql)
    d = curs.fetchall()
    print(d)
    fields = (&#039;bin&#039;, &#039;sum&#039;)
    return jsonify(data=d)


@app.route(&#039;/dataset&#039;, methods=[&#039;POST&#039;])
def dataset():
    # print(request.get_data())
    print(request.files)

    dtrain = request.files[&#039;train&#039;]
    dtest = request.files[&#039;test&#039;]

    #Save input data files in input folder
    dtrain.save(&quot;input/&quot; + dtrain.filename)
    dtest.save(&quot;input/&quot; + dtest.filename)

    df_train = pd.read_csv(&quot;input/&quot; + dtrain.filename)
    # print(df_train.head())

    df_test = pd.read_csv(&quot;input/&quot; + dtest.filename)
    # print(df_test.head())

    #From Jason&#039;s ML module
    X, y = utils.X_y_split(X_train=df_train, X_test=df_test)
    model = fa.All()
    model.fit(X, y)

    #Append prediction column to test set
    predictions = model.predict(df_test)
    df_test[&#039;prediction&#039;] = predictions

    #Save basic stats in output folder
    if model.classification:
        model_class=&quot;Classification&quot;
    else:
        model_class=&quot;Regression&quot;
    bstats={&#039;Model class&#039; : model_class,
            &#039;N obs, train&#039; : df_train.shape[0],
            &#039;N obs, test&#039; : df_test.shape[0],
            &#039;N features&#039; : df_train.shape[1],
            &#039;Metric score&#039; : model.score,
            &#039;Model metric&#039; : model.score_type}
    df_bstats=pd.DataFrame.from_dict(bstats, orient=&quot;index&quot;)
    global json_bstats
    json_bstats = json.dumps(bstats, sort_keys=True,
                             indent=4, separators=(&#039;,&#039;, &#039;: &#039;))
    print(json_bstats)
    df_bstats.to_csv(&quot;output/&quot; + &quot;bstats.csv&quot;, index=True)

    #Save prediction in output folder
    print(df_test.head())
    df_test.to_csv(&quot;output/&quot; + &quot;prediction.csv&quot;, index=False)
    print(&quot;%s: %.3f (%s)&quot; % (&quot;Jacky&#039;s data:&quot;, model.score, model.score_type))
    return &#039;{ &quot;fake_json&quot;:100}&#039;, 200


@app.route(&#039;/prediction&#039;)
def download(filename=None):
    # uploads = os.path.join(current_app.root_path, app.config[&#039;UPLOAD_FOLDER&#039;])
    return send_from_directory(directory=os.path.abspath(os.path.join(&#039;../flask/output&#039;)), filename=&quot;prediction.csv&quot;)
    # return &#039;{ &quot;fake_json&quot;:100}&#039;

@app.route(&#039;/bstats&#039;)
def bstats(filename=None):
    return (json_bstats)
</pre>

		<p class="file_page_meta no_print" style="line-height: 1.5rem;">
			<label class="checkbox normal mini float_right no_top_padding no_min_width">
				<input type="checkbox" id="file_preview_wrap_cb"> wrap long lines			</label>
		</p>

	</div>

	<div id="comments_holder" class="clearfix clear_both">
	<div class="col span_1_of_6"></div>
	<div class="col span_4_of_6 no_right_padding">
		<div id="file_page_comments">
					</div>	
		

	<form action="https://ucbischool.slack.com/files/jzhang/F6DMT5DMJ/viz.py"
			id="file_comment_form"
							class="comment_form clearfix"
						method="post">
					<a href="/team/arvin.sahni" class="member_preview_link" data-member-id="U1V3NSQ6N" >
				<span class="member_image thumb_36" style="background-image: url('https://avatars.slack-edge.com/2017-01-12/127270365911_9d544bb6f6d96cbac129_72.jpg')" data-thumb-size="36" data-member-id="U1V3NSQ6N"></span>
			</a>
				<input type="hidden" name="addcomment" value="1" />
		<input type="hidden" name="crumb" value="s-1501032381-df0fd4d257-☃" />

					<div id="file_comment" class="small texty_comment_input comment_input small_bottom_margin" name="comment" wrap="virtual" ></div>
				<span class="input_note float_left indifferent_grey file_comment_tip">shift+enter to add a new line</span>		<button id="file_comment_submit_btn" type="submit" class="btn float_right  ladda-button" data-style="expand-right"><span class="ladda-label">Add Comment</span></button>
	</form>

<form
		id="file_edit_comment_form"
					class="edit_comment_form clearfix hidden"
				method="post">
				<div id="file_edit_comment" class="small texty_comment_input comment_input small_bottom_margin" name="comment" wrap="virtual"></div>
		<input type="submit" class="save btn float_right " value="Save" />
	<button class="cancel btn btn_outline float_right small_right_margin ">Cancel</button>
</form>	
	</div>
	<div class="col span_1_of_6"></div>
</div>
</div>



		
	</div>
	<div id="overlay"></div>
</div>







<script type="text/javascript">

	/**
	 * A placeholder function that the build script uses to
	 * replace file paths with their CDN versions.
	 *
	 * @param {String} file_path - File path
	 * @returns {String}
	 */
	function vvv(file_path) {

		var vvv_warning = 'You cannot use vvv on dynamic values. Please make sure you only pass in static file paths.';
		if (TS && TS.warn) {
			TS.warn(vvv_warning);
		} else {
			console.warn(vvv_warning);
		}

		return file_path;
	}

	var cdn_url = "https:\/\/slack.global.ssl.fastly.net";
	var vvv_abs_url = "https:\/\/slack.com\/";
	var inc_js_setup_data = {
			emoji_sheets: {
			apple: 'https://a.slack-edge.com/bfaba/img/emoji_2016_06_08/sheet_apple_64_indexed_256colors.png',
			google: 'https://a.slack-edge.com/f360/img/emoji_2016_06_08/sheet_google_64_indexed_128colors.png',
			twitter: 'https://a.slack-edge.com/bfaba/img/emoji_2016_06_08/sheet_twitter_64_indexed_128colors.png',
			emojione: 'https://a.slack-edge.com/bfaba/img/emoji_2016_06_08/sheet_emojione_64_indexed_128colors.png',
		},
		};
</script>
			<script type="text/javascript">
<!--
	// common boot_data
	var boot_data = {
		start_ms: Date.now(),
		app: 'web',
		user_id: 'U1V3NSQ6N',
		team_id: 'T0WA5NWKG',
		no_login: false,
		version_ts: '1501032182',
		version_uid: 'fd0305e179bf52d78589f4b8e7dcc2476d89cc13',
		cache_version: "v16-giraffe",
		cache_ts_version: "v2-bunny",
		redir_domain: 'slack-redir.net',
		signin_url: 'https://slack.com/signin',
		abs_root_url: 'https://slack.com/',
		api_url: '/api/',
		team_url: 'https://ucbischool.slack.com/',
		image_proxy_url: 'https://slack-imgs.com/',
		beacon_timing_url: "https:\/\/slack.com\/beacon\/timing",
		beacon_error_url: "https:\/\/slack.com\/beacon\/error",
		clog_url: "clog\/track\/",
		api_token: 'xoxs-30345778662-63124908226-107418598663-98185ee74f',
		ls_disabled: false,
		use_react_sidebar: false,

		vvv_paths: {
			
		lz_string: "https:\/\/a.slack-edge.com\/bv1-1\/lz-string-1.4.4.worker.8de1b00d670ff3dc706a0.js",
		codemirror: "https:\/\/a.slack-edge.com\/bv1-1\/codemirror.4220b38018a41f354787.min.js",
	codemirror_addon_simple: "https:\/\/a.slack-edge.com\/bv1-1\/simple.3de0be1bda7ab93e8566.min.js",
	codemirror_load: "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_load.b8bbf1b90d31cda997ac.min.js",

	// Silly long list of every possible file that Codemirror might load
	codemirror_files: {
		'apl': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_apl.ca948ff6335149a75e2c.min.js",
		'asciiarmor': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_asciiarmor.6a6aaf06b41b5ebf5320.min.js",
		'asn.1': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_asn.1.c954d9bc29cbf5dc21cc.min.js",
		'asterisk': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_asterisk.8a7fd522a7d398f9433d.min.js",
		'brainfuck': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_brainfuck.7335235606d5082932b7.min.js",
		'clike': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_clike.13f7b214b553c7dbc5ce.min.js",
		'clojure': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_clojure.d8eba78b157273aff3e6.min.js",
		'cmake': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_cmake.3233452d9698f129702e.min.js",
		'cobol': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_cobol.c75d984c48fc8228d0b9.min.js",
		'coffeescript': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_coffeescript.145ebecf51b228d3ed8a.min.js",
		'commonlisp': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_commonlisp.35b7613298b05eb03d4f.min.js",
		'css': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_css.35f1403067c6d3bfdbac.min.js",
		'crystal': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_crystal.fc22c9462faae3abf7ae.min.js",
		'cypher': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_cypher.7803cc048c83fedd7d68.min.js",
		'd': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_d.474811e4a4ce7db146cc.min.js",
		'dart': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_dart.2ae8107b04cb371ebed3.min.js",
		'diff': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_diff.18407e5cfd09e349f8aa.min.js",
		'django': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_django.12a926b6ca525958d5df.min.js",
		'dockerfile': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_dockerfile.c64e844828ba46568004.min.js",
		'dtd': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_dtd.1ab705e774f80aad0e8c.min.js",
		'dylan': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_dylan.388bebb3b2978bc2a37b.min.js",
		'ebnf': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_ebnf.42e478a559030def23c1.min.js",
		'ecl': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_ecl.78dbe5845624d42bf930.min.js",
		'eiffel': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_eiffel.b4e5b4de60dc9ad25073.min.js",
		'elm': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_elm.aaa8d5da021c11541dcf.min.js",
		'erlang': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_erlang.3d15e45d1445a87e5c62.min.js",
		'factor': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_factor.04a4aa32da8491c8b76d.min.js",
		'forth': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_forth.64644bb6fffc973dbde9.min.js",
		'fortran': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_fortran.643e90e26343cf761e66.min.js",
		'gas': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_gas.ce505f267e8dcbab28d3.min.js",
		'gfm': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_gfm.02ff7447f08687e778e9.min.js",
		'gherkin': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_gherkin.d959b75782bcd8868bc0.min.js",
		'go': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_go.6d8bf71ff30c34b5a629.min.js",
		'groovy': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_groovy.a6b9d6ac39727408a51e.min.js",
		'haml': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_haml.112498453e0068bdae4b.min.js",
		'handlebars': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_handlebars.ebdd800c01068472ad5a.min.js",
		'haskell': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_haskell.14049734ac0bcca958ef.min.js",
		'haxe': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_haxe.1cff797be8c121027b54.min.js",
		'htmlembedded': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_htmlembedded.9500117c95cfd15fa0c8.min.js",
		'htmlmixed': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_htmlmixed.0b3ac3b6db16d34ab608.min.js",
		'http': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_http.38538fcfaa943a9ad19b.min.js",
		'idl': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_idl.b3ecf0a360a77d36d1a0.min.js",
		'jade': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_jade.e3c3dddb73bf42eb7929.min.js",
		'javascript': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_javascript.b6f57188464c8adfe7f7.min.js",
		'jinja2': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_jinja2.57c53da5a20a62e3b2e6.min.js",
		'julia': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_julia.73616a2d0fac0acac4bd.min.js",
		'livescript': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_livescript.890cc557fde59fbc7b48.min.js",
		'lua': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_lua.81a94ba3f3a55c3e99a6.min.js",
		'markdown': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_markdown.63a4577bd5f0c5fca7c0.min.js",
		'mathematica': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_mathematica.f7ceb99df3f7d0975715.min.js",
		'mirc': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_mirc.38f46276159d14f9c7da.min.js",
		'mllike': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_mllike.ecf67430bd9f3e6b0b14.min.js",
		'modelica': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_modelica.f17306988d95f18b02b1.min.js",
		'mscgen': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_mscgen.fda581b7271e81bc0f8b.min.js",
		'mumps': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_mumps.fd6454bcd3ae13ffca8b.min.js",
		'nginx': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_nginx.420afbe87f8f52712f4c.min.js",
		'nsis': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_nsis.b742c70fb3d58aad912b.min.js",
		'ntriples': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_ntriples.0eaed0e81a22a769d18d.min.js",
		'octave': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_octave.26b274e9da080ce00ca4.min.js",
		'oz': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_oz.b0da2a413dee9f6a83f8.min.js",
		'pascal': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_pascal.2196e3ed90d3fa5cb770.min.js",
		'pegjs': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_pegjs.11301838188703f27a0b.min.js",
		'perl': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_perl.bad2ff25971d3ad2b6f5.min.js",
		'php': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_php.e76c53b6253f246f59e7.min.js",
		'pig': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_pig.901b8687adddb07fd04e.min.js",
		'powershell': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_powershell.acce3887cc197e04e51b.min.js",
		'properties': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_properties.ec010aefe3c3f00bb241.min.js",
		'puppet': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_puppet.859afe337d2767259298.min.js",
		'python': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_python.277c77d6015d413a89df.min.js",
		'q': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_q.e4f36aa58839ebc47dd6.min.js",
		'r': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_r.3e434a6d36ff097d05d9.min.js",
		'rpm': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_rpm.5df8bb2d6293d37233cb.min.js",
		'rst': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_rst.a36ca1b697ee22a830ad.min.js",
		'ruby': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_ruby.978b9aaf120d295145bc.min.js",
		'rust': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_rust.7981855a2282e070f23a.min.js",
		'sass': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_sass.b04baf68d1d10e680bb2.min.js",
		'scheme': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_scheme.91ca8e3d8a611303da9b.min.js",
		'shell': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_shell.a88b524593250eadfe7e.min.js",
		'sieve': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_sieve.00df3d5350febcb70fbd.min.js",
		'slim': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_slim.e6cf48cddbd2b26bddb0.min.js",
		'smalltalk': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_smalltalk.d70b3c9cb2c66543dcde.min.js",
		'smarty': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_smarty.d9a1d96edbeaaf00bfd7.min.js",
		'solr': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_solr.82a33ddbc4a1db619ecb.min.js",
		'soy': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_soy.a530910c53918f2133e4.min.js",
		'sparql': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_sparql.e9ab4f4ee65d9013ee8e.min.js",
		'spreadsheet': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_spreadsheet.327e708dca1be46a97e8.min.js",
		'sql': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_sql.3bdc555a0efd68671220.min.js",
		'stex': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_stex.739ab0ef05cc4f25c7b9.min.js",
		'stylus': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_stylus.fc862eb018e7bc6bc3f1.min.js",
		'swift': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_swift.73c13f1f3f091ec5c502.min.js",
		'tcl': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_tcl.95b3321599bf2255b5fd.min.js",
		'textile': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_textile.ffc7f28160a4afbd323a.min.js",
		'tiddlywiki': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_tiddlywiki.00a3a5b2f24a2f351193.min.js",
		'tiki': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_tiki.17d447d326268f245b21.min.js",
		'toml': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_toml.8e28a3b1aa99764607c6.min.js",
		'tornado': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_tornado.0e3331b70bf33b990ef3.min.js",
		'troff': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_troff.f36269b2e3a8123dee7d.min.js",
		'ttcn': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_ttcn.a3d7b5c5f18770f7a1d5.min.js",
		'ttcn:cfg': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_ttcn-cfg.628edd1fde5c50d681e0.min.js",
		'turtle': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_turtle.573f8b9e351935d4984a.min.js",
		'twig': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_twig.2430d0afdc87b88d957d.min.js",
		'vb': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_vb.a2ca957b8fe64f9114d3.min.js",
		'vbscript': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_vbscript.c68a70936665f6963523.min.js",
		'velocity': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_velocity.fd4d610490e1be5748bf.min.js",
		'verilog': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_verilog.3825836cab7d72f6e257.min.js",
		'vhdl': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_vhdl.01d1830612b62291e975.min.js",
		'vue': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_vue.3a639e654bd6b7464d29.min.js",
		'xml': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_xml.54b3d3c32c88399c64d7.min.js",
		'xquery': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_xquery.af1fd6ebf10115c8a838.min.js",
		'yaml': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_yaml.4d4d725c7c49297e5ff4.min.js",
		'z80': "https:\/\/a.slack-edge.com\/bv1-1\/codemirror_lang_z80.e74089a4aad7eb7b3071.min.js"
	}		},

		notification_sounds: [{"value":"b2.mp3","label":"Ding","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/b2.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/b2.ogg"},{"value":"animal_stick.mp3","label":"Boing","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/animal_stick.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/animal_stick.ogg"},{"value":"been_tree.mp3","label":"Drop","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/been_tree.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/been_tree.ogg"},{"value":"complete_quest_requirement.mp3","label":"Ta-da","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/complete_quest_requirement.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/complete_quest_requirement.ogg"},{"value":"confirm_delivery.mp3","label":"Plink","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/confirm_delivery.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/confirm_delivery.ogg"},{"value":"flitterbug.mp3","label":"Wow","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/flitterbug.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/flitterbug.ogg"},{"value":"here_you_go_lighter.mp3","label":"Here you go","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/here_you_go_lighter.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/here_you_go_lighter.ogg"},{"value":"hi_flowers_hit.mp3","label":"Hi","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/hi_flowers_hit.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/hi_flowers_hit.ogg"},{"value":"knock_brush.mp3","label":"Knock Brush","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/knock_brush.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/knock_brush.ogg"},{"value":"save_and_checkout.mp3","label":"Whoa!","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/save_and_checkout.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/save_and_checkout.ogg"},{"value":"item_pickup.mp3","label":"Yoink","url":"https:\/\/slack.global.ssl.fastly.net\/7e91\/sounds\/push\/item_pickup.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/item_pickup.ogg"},{"value":"hummus.mp3","label":"Hummus","url":"https:\/\/slack.global.ssl.fastly.net\/7fa9\/sounds\/push\/hummus.mp3","url_ogg":"https:\/\/slack.global.ssl.fastly.net\/46ebb\/sounds\/push\/hummus.ogg"},{"value":"none","label":"None"}],
		alert_sounds: [{"value":"frog.mp3","label":"Frog","url":"https:\/\/slack.global.ssl.fastly.net\/a34a\/sounds\/frog.mp3"}],
		call_sounds: [{"value":"call\/alert_v2.mp3","label":"Alert","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/alert_v2.mp3"},{"value":"call\/incoming_ring_v2.mp3","label":"Incoming ring","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/incoming_ring_v2.mp3"},{"value":"call\/outgoing_ring_v2.mp3","label":"Outgoing ring","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/outgoing_ring_v2.mp3"},{"value":"call\/pop_v2.mp3","label":"Incoming reaction","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/pop_v2.mp3"},{"value":"call\/they_left_call_v2.mp3","label":"They left call","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/they_left_call_v2.mp3"},{"value":"call\/you_left_call_v2.mp3","label":"You left call","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/you_left_call_v2.mp3"},{"value":"call\/they_joined_call_v2.mp3","label":"They joined call","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/they_joined_call_v2.mp3"},{"value":"call\/you_joined_call_v2.mp3","label":"You joined call","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/you_joined_call_v2.mp3"},{"value":"call\/confirmation_v2.mp3","label":"Confirmation","url":"https:\/\/slack.global.ssl.fastly.net\/08f7\/sounds\/call\/confirmation_v2.mp3"}],
		call_sounds_version: "v2",
				default_tz: "America\/Los_Angeles",
		
		feature_tinyspeck: false,
		feature_create_team_google_auth: false,
		feature_flannel_fe: false,
		feature_no_placeholders_in_messages: true,
		feature_webapp_always_collect_initial_time_period_stats: false,
		feature_search_skip_context: false,
		feature_flannel_use_canary_sometimes: false,
		feature_store_members_in_redux: false,
		feature_cross_team_deeplink: false,
		feature_file_threads: false,
		feature_new_broadcast: true,
		feature_message_replies_inline: false,
		feature_subteam_members_diff: false,
		feature_a11y_keyboard_shortcuts: false,
		feature_email_ingestion: false,
		feature_msg_consistency: false,
		feature_sli_channel_priority: false,
		feature_global_nav: false,
		feature_react_sidebar: false,
		feature_attachments_inline: false,
		feature_fix_files: true,
		feature_channel_eventlog_client: true,
		feature_paging_api: false,
		feature_enterprise_frecency: false,
		feature_enterprise_move_channels: true,
		feature_compliance_sku: true,
		feature_i18n_checkout_updates: true,
		feature_dunning: true,
		feature_invoice_dunning: false,
		feature_safeguard_org_retention: true,
		feature_ssi_checkout: false,
		feature_basic_analytics: true,
		feature_team_site_basic_analytics: false,
		feature_free_team_analytics_upsell: false,
		feature_analytics_team_overview: false,
		feature_analytics_csv_exports: false,
		feature_refactor_admin_stats: false,
		feature_guest_invitation_restrictions: false,
		feature_teams_to_workspaces: false,
		feature_analytics_weekly_summary: false,
		feature_channel_alert_words: false,
		feature_shared_channels_beta: false,
		feature_conversations_api: false,
		feature_shared_channels_client: false,
		feature_connecting_shared_channels: false,
		feature_disconnecting_shared_channels: false,
		feature_unified_usergroups: false,
		feature_winssb_beta_channel: false,
		feature_presence_sub: true,
		feature_live_support_free_plan: false,
		feature_slackbot_goes_to_college: false,
		feature_newxp_enqueue_message: true,
		feature_lato_2_ssb: true,
		feature_focus_mode: false,
		feature_star_shortcut: false,
		feature_show_jumper_scores: true,
		feature_force_ls_compression: false,
		feature_zendesk_app_submission_improvement: false,
		feature_ignore_code_mentions: true,
		feature_name_tagging_client: false,
		feature_name_tagging_client_autocomplete: false,
		feature_name_tagging_client_topic_purpose: false,
		feature_use_imgproxy_resizing: true,
		feature_app_forms: false,
		feature_localization: false,
		feature_locale_es_ES: false,
		feature_locale_fr_FR: false,
		feature_locale_de_DE: false,
		feature_locale_ja_JP: false,
		feature_pseudo_locale: false,
		feature_share_mention_comment_cleanup: false,
		feature_external_files: false,
		feature_min_web: true,
		feature_electron_memory_logging: false,
		feature_channel_exports: false,
		feature_prev_next_button: false,
		feature_free_inactive_domains: true,
		feature_keyboard_navigation: true,
		feature_measure_css_usage: false,
		feature_take_profile_photo: false,
		feature_arugula: false,
		feature_texty: true,
		feature_texty_takes_over: true,
		feature_texty_browser_substitutions: false,
		feature_texty_rewrite_on_paste: false,
		feature_texty_search: false,
		feature_more_texty_search: false,
		feature_texty_mentions: false,
		feature_parsed_mrkdwn: false,
		feature_toggle_id_translation: false,
		feature_emoji_menu_tuning: false,
		feature_default_shared_channels: false,
		feature_react_lfs: false,
		feature_channel_options_review_stage: false,
		feature_removing_workspaces_from_cross_workspace_channels: false,
		feature_workspace_request: false,
		feature_disc_workspace_pagination: false,
		feature_enable_mdm: false,
		feature_message_menus: true,
		feature_sli_recaps: true,
		feature_sli_recaps_interface: true,
		feature_slackbot_help_v2: true,
		feature_new_message_click_logging: true,
		feature_user_custom_status: true,
		feature_announce_only_channels: false,
		feature_two_step_oauth: true,
		feature_hide_join_leave: false,
		feature_show_join_leave_pref: true,
		feature_update_emoji_to_v4: false,
		feature_join_leave_rollup_builder: true,
		feature_slim_scrollbar: false,
		feature_i18n_emoji: false,
		feature_sli_briefing: true,
		feature_sli_channel_insights: false,
		feature_sli_file_search: false,
		feature_platform_disable_rtm: true,
		feature_scrollback_half_measures: true,
		feature_initial_scroll_position: true,
		feature_calc_unread_minimums: true,
		feature_notif_prefs_overhaul: true,
		feature_react_messages: false,
		feature_enterprise_loading_state: false,
		feature_api_admin_page: true,
		feature_api_admin_page_not_ent: false,
		feature_imgproxy_multicloud: false,
		feature_app_permissions_api_site: false,
		feature_app_index: false,
		feature_untangle_app_directory_templates: false,
		feature_scim_profile_fields: true,
		feature_ms_msg_handlers_profiling: true,
		feature_ms_latest: false,
		feature_no_preload_video: true,
		feature_react_emoji_picker_frecency: false,
		feature_one_rebuild_per_animation_frame: true,
		feature_app_space: true,
		feature_app_space_permissions_tab: false,
		feature_queue_metrics: false,
		feature_stop_loud_channel_mentions: false,
		feature_message_input_byte_limit: false,
		feature_perfectrics: false,
		feature_automated_perfectrics: false,
		feature_retire_team_site_directory: false,
		feature_link_buttons: false,
		feature_nudge_team_creators: false,
		feature_onedrive_picker: false,
		feature_aaa_app_dir: true,
		feature_quest_onboarding: false,
		feature_less_accessible_id_fetching: true,
		feature_lazy_teams: false,
		feature_delete_team_and_apps: false,
		feature_email_forwarding: false,
		feature_pjpeg: false,
		feature_apps_manage_permissions_scope_changes: false,
		feature_app_dir_only_default_true: false,
		feature_direct_install: false,
		feature_configurable_spellchecker: false,
		feature_last_command_in_tab_complete: false,
		feature_speedy_boot_handlebars: false,
		feature_react_scrollbar: false,
		feature_i18n_channel_meta: false,
		feature_i18n_reactions: false,
		feature_i18n_app_space_im_menu: false,

	client_logs: {"0":{"numbers":[0],"user_facing":false},"@scott":{"numbers":[2,4,37,58,67,141,142,389,481,488,529,667,773,808,888,999],"owner":"@scott"},"@eric":{"numbers":[2,23,47,48,72,73,82,91,93,96,365,438,552,777,794],"owner":"@eric"},"2":{"owner":"@scott \/ @eric","numbers":[2],"user_facing":false},"4":{"owner":"@scott","numbers":[4],"user_facing":false},"5":{"channels":"#dhtml","numbers":[5],"user_facing":false},"23":{"owner":"@eric","numbers":[23],"user_facing":false},"sounds":{"owner":"@scott","name":"sounds","numbers":[37]},"37":{"owner":"@scott","name":"sounds","numbers":[37],"user_facing":true},"47":{"owner":"@eric","numbers":[47],"user_facing":false},"48":{"owner":"@eric","numbers":[48],"user_facing":false},"Message History":{"owner":"@scott","name":"Message History","numbers":[58]},"58":{"owner":"@scott","name":"Message History","numbers":[58],"user_facing":true},"67":{"owner":"@scott","numbers":[67],"user_facing":false},"72":{"owner":"@eric","numbers":[72],"user_facing":false},"73":{"owner":"@eric","numbers":[73],"user_facing":false},"82":{"owner":"@eric","numbers":[82],"user_facing":false},"@shinypb":{"owner":"@shinypb","numbers":[88,1000,1989,1990,1991,1996]},"88":{"owner":"@shinypb","numbers":[88],"user_facing":false},"91":{"owner":"@eric","numbers":[91],"user_facing":false},"93":{"owner":"@eric","numbers":[93],"user_facing":false},"96":{"owner":"@eric","numbers":[96],"user_facing":false},"@steveb":{"owner":"@steveb","numbers":[99]},"99":{"owner":"@steveb","numbers":[99],"user_facing":false},"Channel Marking (MS)":{"owner":"@scott","name":"Channel Marking (MS)","numbers":[141]},"141":{"owner":"@scott","name":"Channel Marking (MS)","numbers":[141],"user_facing":true},"Channel Marking (Client)":{"owner":"@scott","name":"Channel Marking (Client)","numbers":[142]},"142":{"owner":"@scott","name":"Channel Marking (Client)","numbers":[142],"user_facing":true},"365":{"owner":"@eric","numbers":[365],"user_facing":false},"389":{"owner":"@scott","numbers":[389],"user_facing":false},"438":{"owner":"@eric","numbers":[438],"user_facing":false},"@rowan":{"numbers":[444,666],"owner":"@rowan"},"444":{"owner":"@rowan","numbers":[444],"user_facing":false},"481":{"owner":"@scott","numbers":[481],"user_facing":false},"488":{"owner":"@scott","numbers":[488],"user_facing":false},"529":{"owner":"@scott","numbers":[529],"user_facing":false},"552":{"owner":"@eric","numbers":[552],"user_facing":false},"dashboard":{"owner":"@ac \/ @bruce \/ @kylestetz \/ @nic \/ @rowan","channels":"#devel-enterprise-fe, #feat-enterprise-dash","name":"dashboard","numbers":[666]},"@ac":{"channels":"#devel-enterprise-fe, #feat-enterprise-dash","name":"dashboard","numbers":[666],"owner":"@ac"},"@bruce":{"channels":"#devel-enterprise-fe, #feat-enterprise-dash","name":"dashboard","numbers":[666],"owner":"@bruce"},"@kylestetz":{"channels":"#devel-enterprise-fe, #feat-enterprise-dash","name":"dashboard","numbers":[666],"owner":"@kylestetz"},"@nic":{"channels":"#devel-enterprise-fe, #feat-enterprise-dash","name":"dashboard","numbers":[666],"owner":"@nic"},"666":{"owner":"@ac \/ @bruce \/ @kylestetz \/ @nic \/ @rowan","channels":"#devel-enterprise-fe, #feat-enterprise-dash","name":"dashboard","numbers":[666],"user_facing":false},"667":{"owner":"@scott","numbers":[667],"user_facing":false},"773":{"owner":"@scott","numbers":[773],"user_facing":false},"777":{"owner":"@eric","numbers":[777],"user_facing":false},"794":{"owner":"@eric","numbers":[794],"user_facing":false},"Client Responsiveness":{"owner":"@scott","name":"Client Responsiveness","user_facing":false,"numbers":[808]},"808":{"owner":"@scott","name":"Client Responsiveness","user_facing":false,"numbers":[808]},"Message Pane Scrolling":{"owner":"@scott","name":"Message Pane Scrolling","numbers":[888]},"888":{"owner":"@scott","name":"Message Pane Scrolling","numbers":[888],"user_facing":true},"999":{"owner":"@scott","numbers":[999],"user_facing":false},"1000":{"owner":"@shinypb","numbers":[1000],"user_facing":false},"Members":{"owner":"@fearon","name":"Members","numbers":[1975]},"@fearon":{"owner":"@fearon","name":"Members","numbers":[1975,98765]},"1975":{"owner":"@fearon","name":"Members","numbers":[1975],"user_facing":true},"lazy loading":{"owner":"@shinypb","channels":"#devel-flannel","name":"lazy loading","numbers":[1989]},"1989":{"owner":"@shinypb","channels":"#devel-flannel","name":"lazy loading","numbers":[1989],"user_facing":true},"thin_channel_membership":{"owner":"@shinypb","channels":"#devel-infrastructure","name":"thin_channel_membership","numbers":[1990]},"1990":{"owner":"@shinypb","channels":"#devel-infrastructure","name":"thin_channel_membership","numbers":[1990],"user_facing":true},"stats":{"owner":"@shinypb","channels":"#team-infrastructure","name":"stats","numbers":[1991]},"1991":{"owner":"@shinypb","channels":"#team-infrastructure","name":"stats","numbers":[1991],"user_facing":true},"ms":{"owner":"@shinypb","name":"ms","numbers":[1996]},"1996":{"owner":"@shinypb","name":"ms","numbers":[1996],"user_facing":true},"shared_channels_connection":{"owner":"@jim","name":"shared_channels_connection","numbers":[1999]},"@jim":{"owner":"@jim","name":"shared_channels_connection","numbers":[1999]},"1999":{"owner":"@jim","name":"shared_channels_connection","numbers":[1999],"user_facing":false},"dnd":{"owner":"@patrick","channels":"dhtml","name":"dnd","numbers":[2002]},"@patrick":{"owner":"@patrick","channels":"dhtml","name":"dnd","numbers":[2002,2003,2004,2005]},"2002":{"owner":"@patrick","channels":"dhtml","name":"dnd","numbers":[2002],"user_facing":true},"2003":{"owner":"@patrick","channels":"dhtml","numbers":[2003],"user_facing":false},"Threads":{"owner":"@patrick","channels":"#feat-threads, #devel-threads","name":"Threads","numbers":[2004]},"2004":{"owner":"@patrick","channels":"#feat-threads, #devel-threads","name":"Threads","numbers":[2004],"user_facing":true},"2005":{"owner":"@patrick","numbers":[2005],"user_facing":false},"Presence Detection":{"owner":"@ainjii","name":"Presence Detection","numbers":[2017]},"@ainjii":{"owner":"@ainjii","name":"Presence Detection","numbers":[2017,8675309]},"2017":{"owner":"@ainjii","name":"Presence Detection","numbers":[2017],"user_facing":true},"mc_sibs":{"name":"mc_sibs","numbers":[9999]},"9999":{"name":"mc_sibs","numbers":[9999],"user_facing":false},"98765":{"owner":"@fearon","numbers":[98765],"user_facing":false},"8675309":{"owner":"@ainjii","numbers":[8675309],"user_facing":false},"@monty":{"owner":"@monty","numbers":[6532]},"6532":{"owner":"@monty","numbers":[6532],"user_facing":false}},


		img: {
			app_icon: 'https://a.slack-edge.com/272a/img/slack_growl_icon.png'
		},
		page_needs_custom_emoji: false,
		page_needs_team_profile_fields: false,
		page_needs_enterprise: false	};

	
	
	
	
	
	// i18n locale map (eg: maps Slack `ja-jp` to ZD `ja`)
			boot_data.slack_to_zd_locale = {"en-us":"en-us","fr-fr":"fr-fr","de-de":"de","es-es":"es","ja-jp":"ja"};
	
	// client boot data
		
					boot_data.should_use_flannel = true;
		boot_data.ms_connect_url = "wss:\/\/mpmulti-9NCX.slack-msgs.com\/?flannel=1&token=xoxs-30345778662-63124908226-107418598663-98185ee74f";
				boot_data.page_has_incomplete_user_model = true;
		boot_data.flannel_server_pool = "random";
		
	
	
	
	
	
//-->
</script>	
	
	



		


	<!-- output_js "core" -->
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/emoji.d2ee29705f37974e3809.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/rollup-core_required_libs.f18d4ad9eea496f03a13.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/rollup-core_required_ts.e414d993f4bdb8f2421c.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/TS.web.503d03dadda707871841.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>

	<!-- output_js "core_web" -->
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/rollup-core_web.56435693bf2e3cf46fab.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>

	<!-- output_js "secondary" -->
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/rollup-secondary_a_required.6963e15cce45789a7a73.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/rollup-secondary_b_required.07bf1e54273eac829337.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>

	<!-- output_js "application" -->
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/application.bdb97cdf8335faca4fb8.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>

			
	
	<!-- output_js "regular" -->
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/TS.web.comments.c80ee4699a42f88d910b.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/TS.web.file.5b3e5464e5da4b82ef08.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/codemirror.4220b38018a41f354787.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/codemirror_load.b8bbf1b90d31cda997ac.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>
<script type="text/javascript" src="https://a.slack-edge.com/bv1-1/focus-ring.7d25c07d5e9ffb74e869.min.js" crossorigin="anonymous" onload="window._cdn && _cdn.ok(this, arguments)" onerror="window._cdn && _cdn.failed(this, arguments)"></script>


		<script type="text/javascript">
				TS.clog.setTeam('T0WA5NWKG');
				TS.clog.setUser('U1V3NSQ6N');
			</script>

			<script type="text/javascript">
	<!--
		boot_data.page_needs_custom_emoji = true;

		boot_data.file = {"id":"F6DMT5DMJ","created":1501032186,"timestamp":1501032186,"name":"viz.py","title":"viz.py","mimetype":"text\/plain","filetype":"python","pretty_type":"Python","user":"U1V1R4N75","editable":true,"size":4026,"mode":"snippet","is_external":false,"external_type":"","is_public":false,"public_url_shared":false,"display_as_bot":false,"username":"","url_private":"https:\/\/files.slack.com\/files-pri\/T0WA5NWKG-F6DMT5DMJ\/viz.py","url_private_download":"https:\/\/files.slack.com\/files-pri\/T0WA5NWKG-F6DMT5DMJ\/download\/viz.py","permalink":"https:\/\/ucbischool.slack.com\/files\/jzhang\/F6DMT5DMJ\/viz.py","permalink_public":"https:\/\/slack-files.com\/T0WA5NWKG-F6DMT5DMJ-ed1b6c314e","edit_link":"https:\/\/ucbischool.slack.com\/files\/jzhang\/F6DMT5DMJ\/viz.py\/edit","preview":"from __future__ import division\n\nfrom flask import render_template, request, Response, jsonify, send_from_directory\nfrom app import app\n","preview_highlight":"\u003Cdiv class=\"CodeMirror cm-s-default CodeMirrorServer\" oncopy=\"if(event.clipboardData){event.clipboardData.setData('text\/plain',window.getSelection().toString().replace(\/\\u200b\/g,''));event.preventDefault();event.stopPropagation();}\"\u003E\n\u003Cdiv class=\"CodeMirror-code\"\u003E\n\u003Cdiv\u003E\u003Cpre\u003E\u003Cspan class=\"cm-keyword\"\u003Efrom\u003C\/span\u003E \u003Cspan class=\"cm-variable\"\u003E__future__\u003C\/span\u003E \u003Cspan class=\"cm-keyword\"\u003Eimport\u003C\/span\u003E \u003Cspan class=\"cm-variable\"\u003Edivision\u003C\/span\u003E\u003C\/pre\u003E\u003C\/div\u003E\n\u003Cdiv\u003E\u003Cpre\u003E&#8203;\u003C\/pre\u003E\u003C\/div\u003E\n\u003Cdiv\u003E\u003Cpre\u003E\u003Cspan class=\"cm-keyword\"\u003Efrom\u003C\/span\u003E \u003Cspan class=\"cm-variable\"\u003Eflask\u003C\/span\u003E \u003Cspan class=\"cm-keyword\"\u003Eimport\u003C\/span\u003E \u003Cspan class=\"cm-variable\"\u003Erender_template\u003C\/span\u003E, \u003Cspan class=\"cm-variable\"\u003Erequest\u003C\/span\u003E, \u003Cspan class=\"cm-variable\"\u003EResponse\u003C\/span\u003E, \u003Cspan class=\"cm-variable\"\u003Ejsonify\u003C\/span\u003E, \u003Cspan class=\"cm-variable\"\u003Esend_from_directory\u003C\/span\u003E\u003C\/pre\u003E\u003C\/div\u003E\n\u003Cdiv\u003E\u003Cpre\u003E\u003Cspan class=\"cm-keyword\"\u003Efrom\u003C\/span\u003E \u003Cspan class=\"cm-variable\"\u003Eapp\u003C\/span\u003E \u003Cspan class=\"cm-keyword\"\u003Eimport\u003C\/span\u003E \u003Cspan class=\"cm-variable\"\u003Eapp\u003C\/span\u003E\u003C\/pre\u003E\u003C\/div\u003E\n\u003C\/div\u003E\n\u003C\/div\u003E\n","lines":127,"lines_more":122,"preview_is_truncated":true,"channels":[],"groups":[],"ims":["D5RHYLHB6"],"comments_count":0};
		boot_data.file.comments = [];

		

		var g_editor;

		$(function(){

			var wrap_long_lines = !!TS.model.code_wrap_long_lines;

			g_editor = CodeMirror(function(elt){
				var content = document.getElementById("file_contents");
				content.parentNode.replaceChild(elt, content);
			}, {
				value: $('#file_contents').text(),
				lineNumbers: true,
				matchBrackets: true,
				indentUnit: 4,
				indentWithTabs: true,
				enterMode: "keep",
				tabMode: "shift",
				viewportMargin: 10,
				readOnly: true,
				lineWrapping: wrap_long_lines
			});

			// past a certain point, CodeMirror rendering may simply stop working.
			// start having CodeMirror use its Long List View-style scolling at >= max_lines.
			var max_lines = 8192;

			var snippet_lines;

			// determine # of lines, limit height accordingly
			if (g_editor.doc && g_editor.doc.lineCount) {
				snippet_lines = parseInt(g_editor.doc.lineCount());
				var new_height;
				if (snippet_lines) {
					// we want the CodeMirror container to collapse around short snippets.
					// however, we want to let it auto-expand only up to a limit, at which point
					// we specify a fixed height so CM can display huge snippets without dying.
					// this is because CodeMirror works nicely with no height set, but doesn't
					// scale (big file performance issue), and doesn't work with min/max-height -
					// so at some point, we have to set an absolute height to kick it into its
					// smart / partial "Long List View"-style rendering mode.
					if (snippet_lines < max_lines) {
						new_height = 'auto';
					} else {
						new_height = (max_lines * 0.875) + 'rem'; // line-to-rem ratio
					}
					var line_count = Math.min(snippet_lines, max_lines);
					TS.info('Applying height of ' + new_height + ' to container for this snippet of ' + snippet_lines + (snippet_lines === 1 ? ' line' : ' lines'));
					$('#page .CodeMirror').height(new_height);
				}
			}

			$('#file_preview_wrap_cb').bind('change', function(e) {
				TS.model.code_wrap_long_lines = $(this).prop('checked');
				g_editor.setOption('lineWrapping', TS.model.code_wrap_long_lines);
			})

			$('#file_preview_wrap_cb').prop('checked', wrap_long_lines);

			CodeMirror.switchSlackMode(g_editor, "python");
		});

		
		$('#file_comment').css('overflow', 'hidden').autogrow();
	//-->
	</script>


			<script type="text/javascript">TS.boot(boot_data);</script>
	
<style>.color_9f69e7:not(.nuc) {color:#9F69E7;}.color_4bbe2e:not(.nuc) {color:#4BBE2E;}.color_e7392d:not(.nuc) {color:#E7392D;}.color_3c989f:not(.nuc) {color:#3C989F;}.color_674b1b:not(.nuc) {color:#674B1B;}.color_e96699:not(.nuc) {color:#E96699;}.color_e0a729:not(.nuc) {color:#E0A729;}.color_684b6c:not(.nuc) {color:#684B6C;}.color_5b89d5:not(.nuc) {color:#5B89D5;}.color_2b6836:not(.nuc) {color:#2B6836;}.color_99a949:not(.nuc) {color:#99A949;}.color_df3dc0:not(.nuc) {color:#DF3DC0;}.color_4cc091:not(.nuc) {color:#4CC091;}.color_9b3b45:not(.nuc) {color:#9B3B45;}.color_d58247:not(.nuc) {color:#D58247;}.color_bb86b7:not(.nuc) {color:#BB86B7;}.color_5a4592:not(.nuc) {color:#5A4592;}.color_db3150:not(.nuc) {color:#DB3150;}.color_235e5b:not(.nuc) {color:#235E5B;}.color_9e3997:not(.nuc) {color:#9E3997;}.color_53b759:not(.nuc) {color:#53B759;}.color_c386df:not(.nuc) {color:#C386DF;}.color_385a86:not(.nuc) {color:#385A86;}.color_a63024:not(.nuc) {color:#A63024;}.color_5870dd:not(.nuc) {color:#5870DD;}.color_ea2977:not(.nuc) {color:#EA2977;}.color_50a0cf:not(.nuc) {color:#50A0CF;}.color_d55aef:not(.nuc) {color:#D55AEF;}.color_d1707d:not(.nuc) {color:#D1707D;}.color_43761b:not(.nuc) {color:#43761B;}.color_e06b56:not(.nuc) {color:#E06B56;}.color_8f4a2b:not(.nuc) {color:#8F4A2B;}.color_902d59:not(.nuc) {color:#902D59;}.color_de5f24:not(.nuc) {color:#DE5F24;}.color_a2a5dc:not(.nuc) {color:#A2A5DC;}.color_827327:not(.nuc) {color:#827327;}.color_3c8c69:not(.nuc) {color:#3C8C69;}.color_8d4b84:not(.nuc) {color:#8D4B84;}.color_84b22f:not(.nuc) {color:#84B22F;}.color_4ec0d6:not(.nuc) {color:#4EC0D6;}.color_e23f99:not(.nuc) {color:#E23F99;}.color_e475df:not(.nuc) {color:#E475DF;}.color_619a4f:not(.nuc) {color:#619A4F;}.color_a72f79:not(.nuc) {color:#A72F79;}.color_7d414c:not(.nuc) {color:#7D414C;}.color_aba727:not(.nuc) {color:#ABA727;}.color_965d1b:not(.nuc) {color:#965D1B;}.color_4d5e26:not(.nuc) {color:#4D5E26;}.color_dd8527:not(.nuc) {color:#DD8527;}.color_bd9336:not(.nuc) {color:#BD9336;}.color_e85d72:not(.nuc) {color:#E85D72;}.color_dc7dbb:not(.nuc) {color:#DC7DBB;}.color_bc3663:not(.nuc) {color:#BC3663;}.color_9d8eee:not(.nuc) {color:#9D8EEE;}.color_8469bc:not(.nuc) {color:#8469BC;}.color_73769d:not(.nuc) {color:#73769D;}.color_b14cbc:not(.nuc) {color:#B14CBC;}</style>

<!-- slack-www-hhvm-033dd32d49d0ac3e9 / 2017-07-25 18:26:21 / vfd0305e179bf52d78589f4b8e7dcc2476d89cc13 / B:H -->


</body>
</html>